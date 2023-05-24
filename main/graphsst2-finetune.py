# In[Import]
import os
import sys
import argparse
import os.path as osp
import numpy as np

machine = 'server'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCELoss
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from dataset.graphsst2 import load_SeniGraph
from dataset.graphsst2 import get_dataloader
from utils.unbatch import unbatch
from utils.unbatch import unbatch_edge_index
import model

# In[argparse]
parser = argparse.ArgumentParser(description='')
parser.add_argument('--cuda', default = 1, 
                    type = int, help = 'cuda device')
parser.add_argument('--datadir', default = 'data/', 
                    type = str, help = 'directory for datasets.')
parser.add_argument('--epochs', default = 60, 
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
parser.add_argument('--batch_size', default=32, 
                    type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, 
                    type=float, help='learning rate for the predictor')
args = parser.parse_args()
args.seed = eval(args.seed)

save_path = '/finetuned/Graphsst2'

os.makedirs(save_path, exist_ok = True)

# In[Device]
if torch.cuda.is_available():
    device = torch.device('cuda:%d' % args.cuda)
else:
    device = torch.device('cpu')

# In[Dataset]
graphs = load_SeniGraph('data','Graph-SST2')
dataloader = get_dataloader(graphs,args.batch_size)

train_loader = dataloader['train']
val_loader = dataloader['eval'] 
test_loader = dataloader['test']
'''
len(train): 28327 len(eval):3147 len(test):12305
'''

# In[Initialize]
node_in_dim = graphs[0].x.shape[1]
svd_seed = list(np.random.randint(500, size = args.seed_num))
graph_class = 2
task = 'graph'

# In[Model]
pretrained = torch.load('trained/pretrain.pt').to(device)
explainer = pretrained.Explainer
projector = nn.Linear(node_in_dim, explainer.node_in_dim).to(device)
predictor = model.GraphPredictor(explainer.node_in_dim, args.node_hidden_dim,
                                 0, graph_class).to(device)

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

# In[Sparsity]
def sparsity(exp):
    selected = torch.where(exp > 0.5, 1, 0).sum().cpu()
    return selected / exp.shape[0]

# In[Train, Valid, and Save]
epochs = args.epochs
max_exp_loss = float('inf')
max_pred_loss = float('inf')
max_val_loss = float('inf')

for epoch in range(1, epochs + 1):
    train_right = 0
    train_sparsity = []
    train_exp_loss = []
    train_pred_loss=  []
    train_total_loss = []
    for graph_batch in train_loader:
        graph_batch.to(device)

        graph_batch.x = F.relu(projector(graph_batch.x))
        explanation = explainer(graph_batch, epoch, False)
        explanation = explanation / torch.max(explanation)
        prediction = predictor(graph_batch, explanation)
        
        train_right += torch.eq(prediction.argmax(dim = 1), graph_batch.y).sum()
        exp_sparsity = sparsity(explanation)
        
        exp_loss = Exp_loss(explanation)
        pred_loss = Pred_loss(prediction, graph_batch.y)
        total_loss = exp_loss + pred_loss
        
        train_sparsity.append(exp_sparsity)
        train_exp_loss.append(exp_loss.cpu().item())
        train_pred_loss.append(pred_loss.cpu().item())
        train_total_loss.append(total_loss.cpu().item())
        
        opt.zero_grad()
        # pred_loss.backward()
        total_loss.backward()
        opt.step()

    train_acc = train_right / len(train_loader.dataset)
    print('----------------- Training Epoch %d -----------------------' % epoch)
    print('Training predict accuracy:', train_acc.cpu().item())
    print('Training explain sparsity:', np.mean(train_sparsity))
    print('Training explain loss:', np.mean(train_exp_loss),
          '\nTraining predict loss:', np.mean(train_pred_loss),
          '\nTraining total loss:', np.mean(train_total_loss))
    print('-------------------------------------------------------\n')
        
    if epoch % 2 == 0:
        val_right = 0
        val_sparsity = []
        val_exp_loss = []
        val_pred_loss=  []
        val_total_loss = []
        with torch.no_grad():
            for graph_batch in val_loader:
                graph_batch.to(device)

                graph_batch.x = F.relu(projector(graph_batch.x))
                explanation = explainer(graph_batch, 0, False)
                explanation = explanation / torch.max(explanation)
                prediction = predictor(graph_batch, explanation)
                
                val_right += torch.eq(prediction.argmax(dim = 1), graph_batch.y).sum()
                exp_sparsity = sparsity(explanation)
                    
                exp_loss = Exp_loss(explanation)
                pred_loss = Pred_loss(prediction, graph_batch.y)
                total_loss = exp_loss + pred_loss

                val_sparsity.append(exp_sparsity)    
                val_exp_loss.append(exp_loss.item())
                val_pred_loss.append(pred_loss.item())
                val_total_loss.append(total_loss.item())
                    
            if max_val_loss > np.mean(val_total_loss):
                max_val_loss = np.mean(val_total_loss)
                torch.save(explainer.cpu(), osp.join('finetuned/Graphsst2/Explainer.pt'))
                torch.save(projector.cpu(), osp.join('finetuned/Graphsst2/Projector.pt'))
                torch.save(predictor.cpu(), osp.join('finetuned/Graphsst2/Predictor.pt'))
                explainer.to(device)
                projector.to(device)
                predictor.to(device)

                if max_exp_loss > np.mean(val_exp_loss):
                    max_exp_loss = np.mean(val_exp_loss)
                    print('Explain well.')
                if max_pred_loss > np.mean(val_pred_loss):
                    max_pred_loss = np.mean(val_pred_loss)
                    print('Predict well.')

            val_acc = val_right / len(val_loader.dataset)
            print('----------------- Validation Epoch %d -----------------------' % epoch)
            print('Validation predict accuray:', val_acc.cpu().item())
            print('Validation explain sparsity:', np.mean(val_sparsity))
            print('Validation explain loss:', np.mean(val_exp_loss),
                  '\nValidation predict loss:', np.mean(val_pred_loss),
                  '\nValidation total loss:', np.mean(val_total_loss))
            print('-------------------------------------------------------\n')

# In[Test]
best_explainer = torch.load('finetuned/Graphsst2/Explainer.pt').to(device)
best_projector = torch.load('finetuned/Graphsst2/Projector.pt').to(device)
best_predictor = torch.load('finetuned/Graphsst2/Predictor.pt').to(device)
test_right = 0
test_sparsity = []
test_exp_loss = []
test_pred_loss=  []
test_total_loss = []
with torch.no_grad():
    for graph_batch in test_loader:
        graph_batch.to(device)

        graph_batch.x = F.relu(best_projector(graph_batch.x))
        explanation = best_explainer(graph_batch, 0, False)
        explanation = explanation / torch.max(explanation)
        prediction = best_predictor(graph_batch, explanation)
        
        test_right += torch.eq(prediction.argmax(dim = 1), graph_batch.y).sum()
        exp_sparsity = sparsity(explanation)
        
        exp_loss = Exp_loss(explanation)
        pred_loss = Pred_loss(prediction, graph_batch.y)
        total_loss = exp_loss + pred_loss
        
        test_sparsity.append(exp_sparsity)
        test_exp_loss.append(exp_loss.item())
        test_pred_loss.append(pred_loss.item())
        test_total_loss.append(total_loss.item())

    test_acc = test_right / len(test_loader.dataset)
    print('----------------- Testing -----------------------')
    print('Testing predict accuracy:', test_acc.cpu().item())
    print('Testing explain sparsity:', np.mean(test_sparsity))
    print('Testing explain loss:', np.mean(test_exp_loss),
          '\nTesting predict loss:', np.mean(test_pred_loss),
          '\nTesting total loss:', np.mean(test_total_loss))
    print('-------------------------------------------------------\n')
