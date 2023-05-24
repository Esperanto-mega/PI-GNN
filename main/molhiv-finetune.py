# In[Import]
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import argparse
import os.path as osp
import numpy as np
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCELoss
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score
from ogb.graphproppred import Evaluator

from utils.unbatch import unbatch
from utils import mask
import model

# In[argparse]
parser = argparse.ArgumentParser(description='')
parser.add_argument('--cuda', default = 3, 
                    type = int, help = 'cuda device')
parser.add_argument('--datadir', default = 'data/', 
                    type = str, help = 'directory for datasets.')
parser.add_argument('--epochs', default = 40, 
                    type = int, help = 'training iterations')
parser.add_argument('--seed',  nargs = '?', 
                    default = '[1,2,3]', help = 'random seed')
# hyper 
parser.add_argument('--node_hidden_dim', default = 64,
                    type = int, help = 'hidden dimension of node feature')
parser.add_argument('--svd_dim', default = 16,
                    type = int, help = 'embedding dimension of svd result')
parser.add_argument('--e2r_hidden_dim', default = 32,
                    type = int, help = 'hidden representation dimension')
parser.add_argument('--e2r_out_dim', default = 16,
                    type = int, help = 'out dimension of structural representation')
parser.add_argument('--hyper_dim', default = 64,
                    type = int, help = 'hyper-node (raw-edge) representaion dimension')
parser.add_argument('--dif_ratio', default = 0.25,
                    type = float, help = 'ratio of diffusion on raw embedding')
parser.add_argument('--dif_decay', default = 0.999,
                    type = float, help = 'decay coefficient of diffusion')
parser.add_argument('--seed_num',default = 8,
                    type = int, help = 'number of svd random seed')
parser.add_argument('--svd_iter', default = 5,
                    type = int, help = 'number of svd iteration')
# basic
parser.add_argument('--batch_size', default=128, 
                    type=int, help='batch size')
parser.add_argument('--lr', default=1e-3, 
                    type=float, help='learning rate for the predictor')
args = parser.parse_args()
args.seed = eval(args.seed)

save_path = '/finetuned/Molhiv'

os.makedirs(save_path, exist_ok = True)

# In[Device]
if torch.cuda.is_available():
    device = torch.device('cuda:%d' % args.cuda)
else:
    device = torch.device('cpu')

# In[Dataset]
from ogb.graphproppred import PygGraphPropPredDataset

dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv') 
split_idx = dataset.get_idx_split()

train_loader = DataLoader(dataset[split_idx['train']], args.batch_size, True)
val_loader = DataLoader(dataset[split_idx['valid']], args.batch_size, True)
test_loader = DataLoader(dataset[split_idx['test']], args.batch_size, True)

# In[Initialize]
node_in_dim = dataset[0].x.shape[1]
svd_seed = list(np.random.randint(500, size = args.seed_num))
graph_class = 2
task = 'graph'

# In[Predictor]
class MolhivPredictor(torch.nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, edge_in_dim = 0, classes = 10):
        super().__init__()

        mlp1 = nn.Linear(node_in_dim, node_hidden_dim)
        self.gin1 = GINConv(mlp1, train_eps = True)
        mlp2 = nn.Linear(node_hidden_dim, node_hidden_dim)
        self.gin2 = GINConv(mlp2, train_eps = True)
        if edge_in_dim == 0:
            self.predictor = nn.Linear(node_hidden_dim, classes)
        else:
            self.predictor = nn.Linear(node_hidden_dim + edge_in_dim, classes)

    def forward(self, g, weights, edge_pool = None):
        mask.set_mask(weights, self.gin1)
        z = F.relu(self.gin1(g.x, g.edge_index))
        mask.clear_mask(self.gin1)

        mask.set_mask(weights, self.gin2)
        z = F.relu(self.gin2(z, g.edge_index))
        mask.clear_mask(self.gin2)

        global_z = global_mean_pool(z, g.batch)
        if edge_pool is not None:
            global_z = torch.cat([global_z, edge_pool], dim = 1)
        logits = torch.sigmoid(self.predictor(global_z))
        return logits

# In[Model]
pretrained = torch.load('trained/pretrain.pt')
explainer = pretrained.Explainer.to(device)
projector = nn.Linear(node_in_dim, explainer.node_in_dim).to(device)
predictor = MolhivPredictor(explainer.node_in_dim, 
                            args.node_hidden_dim, 
                            explainer.e2r_out_dim,
                            graph_class).to(device)
evaluator = Evaluator('ogbg-molhiv')

# In[Loss]
Pred_loss = CrossEntropyLoss()

def Exp_loss(exp, pi = 0.7):
    EntropyLoss = -(exp * torch.log((exp + 1e-6) / pi) +\
                    (1 - exp) * torch.log((1 - exp + 1e-6) / (1 - pi)))
    EntropyLoss = torch.mean(EntropyLoss)
    return EntropyLoss

# In[Optimizer]
params_dict = [{'params': explainer.parameters(), 'lr': args.lr},
               {'params': predictor.parameters(), 'lr': args.lr},
               {'params': projector.parameters(),'lr': args.lr}]
opt = torch.optim.Adam(params_dict)

def sparsity(exp):
    selected = torch.where(exp > 0.5, 1, 0).sum().cpu()
    return selected / exp.shape[0]

# In[Train, Valid, and Save]
epochs = args.epochs
max_exp_loss = float('inf')
max_pred_loss = float('inf')
max_val_loss = float('inf')

for epoch in range(1, epochs + 1):
    train_roc_auc = []
    train_sparsity = []
    train_exp_loss = []
    train_pred_loss=  []
    train_total_loss = []
    for graph_batch in train_loader:
        graph_batch.to(device)

        graph_batch.x = F.relu(projector(graph_batch.x.float()))
        explanation, edge_pool = explainer(graph_batch, epoch)
        explanation = explanation / torch.max(explanation)
        prediction = predictor(graph_batch, explanation, edge_pool)

        try:
            roc_auc = evaluator.eval({'y_true': graph_batch.y,
                                      'y_pred': prediction.argmax(dim = 1).unsqueeze(dim = 1)})['rocauc']
            # roc_auc = roc_auc_score(graph_batch.y.squeeze(dim = 1).detach().cpu().numpy(),
            #                         prediction.argmax(dim = 1).detach().cpu().numpy())
        except RuntimeError:
            print('Epoch %d: No positively labeled data available. Cannot compute ROC-AUC.' % epoch)
        else:
            # print('Epoch %d roc_auc:' % epoch, roc_auc)
            train_roc_auc.append(roc_auc)

        exp_sparsity = sparsity(explanation)
        exp_loss = Exp_loss(explanation)
        pred_loss = Pred_loss(prediction, graph_batch.y.squeeze(dim = 1))
        total_loss = exp_loss + pred_loss

        train_sparsity.append(exp_sparsity)
        train_exp_loss.append(exp_loss.cpu().item())
        train_pred_loss.append(pred_loss.cpu().item())
        train_total_loss.append(total_loss.cpu().item())
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()
    
    print('----------------- Training Epoch %d -----------------------' % epoch)
    print('Training predict roc-auc:', np.mean(train_roc_auc))
    print('Training explain sparsity:', np.mean(train_sparsity))
    print('Training explain loss:', np.mean(train_exp_loss),
          '\nTraining predict loss:', np.mean(train_pred_loss),
          '\nTraining total loss:', np.mean(train_total_loss))
    print('-------------------------------------------------------\n')
        
    if epoch % 2 == 0:
        val_roc_auc = []
        val_sparsity = []
        val_exp_loss = []
        val_pred_loss=  []
        val_total_loss = []
        with torch.no_grad():
            for graph_batch in val_loader:
                graph_batch.to(device)

                graph_batch.x = F.relu(projector(graph_batch.x.float()))
                explanation, edge_pool = explainer(graph_batch, t = 0)
                explanation = explanation / torch.max(explanation)
                prediction = predictor(graph_batch, explanation, edge_pool)
                
                try:
                    roc_auc = evaluator.eval({'y_true': graph_batch.y,
                                      'y_pred': prediction.argmax(dim = 1).unsqueeze(dim = 1)})['rocauc']
                    # roc_auc = roc_auc_score(graph_batch.y.squeeze(dim = 1).detach().cpu().numpy(),
                    #                 prediction.argmax(dim = 1).detach().cpu().numpy())
                except RuntimeError:
                    print('Epoch %d: No positively labeled data available. Cannot compute ROC-AUC.' % epoch)
                else:
                    # print('Epoch %d roc_auc:' % epoch, roc_auc)
                    val_roc_auc.append(roc_auc)
                    
                exp_sparsity = sparsity(explanation)
                exp_loss = Exp_loss(explanation)
                pred_loss = Pred_loss(prediction, graph_batch.y.squeeze(dim = 1))
                total_loss = exp_loss + pred_loss

                val_sparsity.append(exp_sparsity)
                val_exp_loss.append(exp_loss.item())
                val_pred_loss.append(pred_loss.item())
                val_total_loss.append(total_loss.item())
                    
            if max_val_loss > np.mean(val_total_loss):
                max_val_loss = np.mean(val_total_loss)
                torch.save(explainer.cpu(), osp.join('finetuned/Molhiv/Explainer.pt'))
                torch.save(projector.cpu(), osp.join('finetuned/Molhiv/Projector.pt'))
                torch.save(predictor.cpu(), osp.join('finetuned/Molhiv/Predictor.pt'))
                explainer.to(device)
                projector.to(device)
                predictor.to(device)

                if max_exp_loss > np.mean(val_exp_loss):
                    max_exp_loss = np.mean(val_exp_loss)
                    print('Explain well.')
                if max_pred_loss > np.mean(val_pred_loss):
                    max_pred_loss = np.mean(val_pred_loss)
                    print('Predict well.')

            print('----------------- Validation Epoch %d -----------------------' % epoch)
            print('Validation predict roc-auc:', np.mean(val_roc_auc))
            print('Validation explain sparsity:', np.mean(val_sparsity))
            print('Validation explain loss:', np.mean(val_exp_loss),
                  '\nValidation predict loss:', np.mean(val_pred_loss),
                  '\nValidation total loss:', np.mean(val_total_loss))
            print('-------------------------------------------------------\n')

# In[Test]
best_explainer = torch.load('finetuned/Molhiv/Explainer.pt').to(device)
best_projector = torch.load('finetuned/Molhiv/Projector.pt').to(device)
best_predictor = torch.load('finetuned/Molhiv/Predictor.pt').to(device)
test_roc_auc = []
test_sparsity = []
test_exp_loss = []
test_pred_loss=  []
test_total_loss = []
with torch.no_grad():
    for graph_batch in test_loader:
        graph_batch.to(device)

        graph_batch.x = F.relu(best_projector(graph_batch.x.float()))
        explanation, edge_pool = best_explainer(graph_batch, t = 0)
        explanation = explanation / torch.max(explanation)
        prediction = best_predictor(graph_batch, explanation, edge_pool)
        
        try:
            roc_auc = evaluator.eval({'y_true': graph_batch.y,
                                      'y_pred': prediction.argmax(dim = 1).unsqueeze(dim = 1)})['rocauc']
            # roc_auc = roc_auc_score(graph_batch.y.squeeze(dim = 1).detach().cpu().numpy(),
            #                         prediction.argmax(dim = 1).detach().cpu().numpy())
        except RuntimeError:
            print('Epoch %d: No positively labeled data available. Cannot compute ROC-AUC.' % epoch)
        else:
            # print('Epoch %d roc_auc:' % epoch, roc_auc)
            test_roc_auc.append(roc_auc)

        exp_sparsity = sparsity(explanation)
        exp_loss = Exp_loss(explanation)
        pred_loss = Pred_loss(prediction, graph_batch.y.squeeze(dim = 1))
        total_loss = exp_loss + pred_loss
        
        test_sparsity.append(exp_sparsity)
        test_exp_loss.append(exp_loss.item())
        test_pred_loss.append(pred_loss.item())
        test_total_loss.append(total_loss.item())

    print('----------------- Testing -----------------------')
    print('Testing predict roc-auc:', np.mean(test_roc_auc))
    print('Testing explain sparsity:', np.mean(test_sparsity))
    print('Testing explain loss:', np.mean(test_exp_loss),
          '\nTesting predict loss:', np.mean(test_pred_loss),
          '\nTesting total loss:', np.mean(test_total_loss))
    print('-------------------------------------------------------\n')
