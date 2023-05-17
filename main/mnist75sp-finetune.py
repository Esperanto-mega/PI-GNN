# In[Import]
import os
import sys
import argparse
import os.path as osp
import numpy as np

machine = 'server'

if machine != 'pc':
    cur_path = '/home/data/yj/Bnova'
    if cur_path not in sys.path:
        sys.path.append(cur_path)
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCELoss
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score

from dataset.mnist75sp import MNIST75sp
from utils import mask
import model

# In[argparse]
parser = argparse.ArgumentParser(description='')
parser.add_argument('--cuda', default = 0, 
                    type = int, help = 'cuda device')
parser.add_argument('--datadir', default = 'data/', 
                    type = str, help = 'directory for datasets.')
parser.add_argument('--epochs', default = 30, 
                    type = int, help = 'training iterations')
parser.add_argument('--seed',  nargs = '?', 
                    default = '[1,2,3]', help = 'random seed')
# hyper 
parser.add_argument('--node_hidden_dim', default = 64,
                    type = int, help = 'hidden dimension of node feature')
parser.add_argument('--svd_dim', default = 64,
                    type = int, help = 'embedding dimension of svd result')
parser.add_argument('--e2r_hidden_dim', default = 128,
                    type = int, help = 'hidden representation dimension')
parser.add_argument('--e2r_out_dim', default = 128,
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
parser.add_argument('--batch_size', default=256, 
                    type=int, help='batch size')
parser.add_argument('--lr', default=1e-3, 
                    type=float, help='learning rate for the predictor')
args = parser.parse_args()
args.seed = eval(args.seed)

if machine == 'pc':
    save_path = '/finetuned/Mnist'
else:
    save_path = '/home/data/yj/Bnova/finetuned/Mnist'

os.makedirs(save_path, exist_ok = True)

# In[Device]
if torch.cuda.is_available():
    device = torch.device('cuda:%d' % args.cuda)
else:
    device = torch.device('cpu')

# In[Dataset]
n_train = 20000
n_valid = 5000

''' train-mode: 60000 images, test-mode: 10000 images '''
train_valid = MNIST75sp(osp.join(args.datadir, 'MNIST-75sp/'), mode='train')
train_dataset, val_dataset = train_valid[:n_train], train_valid[-n_valid:]
test_dataset = MNIST75sp(osp.join(args.datadir, 'MNIST-75sp/'), mode='test')

print('len(train_valid):', len(train_valid),
      '; len(test_dataset):', len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False)

# In[Initialize]
node_in_dim = train_dataset[0].x.shape[1]
svd_seed = list(np.random.randint(500, size = args.seed_num))
graph_class = 10
task = 'graph'

# In[Predictor]
class MnistPredictor(torch.nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, classes = 10):
        super().__init__()

        mlp1 = nn.Linear(node_in_dim, node_hidden_dim)
        self.gin1 = GINConv(mlp1, train_eps = True)
        mlp2 = nn.Linear(node_hidden_dim, node_hidden_dim)
        self.gin2 = GINConv(mlp2, train_eps = True)
        self.predictor = nn.Linear(node_hidden_dim, classes)

    def forward(self, g, weights):
        mask.set_mask(weights, self.gin1)
        z = F.relu(self.gin1(g.x, g.edge_index))
        mask.clear_mask(self.gin1)

        mask.set_mask(weights, self.gin2)
        z = F.relu(self.gin2(z, g.edge_index))
        mask.clear_mask(self.gin2)

        global_z = global_mean_pool(z, g.batch)
        logits = torch.sigmoid(self.predictor(global_z))
        return logits

# In[Pretrained]
model = torch.load('trained/pretrain.pt').to(device)
explainer = model.Explainer
projector = nn.Linear(node_in_dim, explainer.node_in_dim).to(device)
predictor = MnistPredictor(explainer.node_in_dim, args.node_hidden_dim).to(device)

# In[Loss]
Pred_loss = CrossEntropyLoss()
Exp_loss = BCELoss()

# In[Optimizer]
params_dict = [{'params': explainer.parameters(), 'lr': 4e-4},
               {'params': predictor.parameters(), 'lr': 4e-4},
               {'params': projector.parameters(),'lr': 4e-4}]
opt = torch.optim.Adam(params_dict)

# In[Explantion acc]
def exp_metric(exp, gt):
    exp = torch.where(exp > 0.5, 1, 0)
    eq = torch.eq(exp, gt).sum()
    acc = eq / len(gt)
    add = exp + gt
    sub = exp - gt
    tp = torch.where(add == 2, 1, 0).sum()
    # tn = torch.where(add == 0, 1, 0).sum()
    fp = torch.where(sub == 1, 1, 0).sum()
    fn = torch.where(sub == -1, 1, 0).sum()
    # print('tp:', tp, 'fp:', fp, 'fn:', fn)
    pr = tp / (tp + fp + 1e-8)
    re = tp / (tp + fn + 1e-8)
    # print('pr:', pr, 're:', re)

    num_gt = int(gt.sum())
    exp_id = exp.sort(descending = True)[1]
    exp_id = exp_id[0:num_gt].detach().cpu().numpy()
    dir_pr = gt[exp_id].sum() / num_gt

    roc_auc = roc_auc_score(gt.detach().cpu().numpy(),exp.detach().cpu().numpy())

    return pr, re, acc, dir_pr, roc_auc

# In[Train, Valid, and Save]
epochs = args.epochs
max_exp_loss = float('inf')
max_pred_loss = float('inf')
max_val_loss = float('inf')

for epoch in range(1, epochs + 1):
    train_right = 0
    train_exp_pr = []
    train_exp_re = []
    train_exp_acc = []
    train_dir_pr = []
    train_roc_auc = []
    train_exp_loss = []
    train_pred_loss=  []
    train_total_loss = []
    for graph_batch in train_loader:
        graph_batch.to(device)

        graph_batch.x = F.relu(projector(graph_batch.x))
        # print('graph_batch.x.shape:',graph_batch.x.shape)
        # print('graph_batch.edge_index.shape:',graph_batch.edge_index.shape)
        explanation = explainer(graph_batch, epoch, False)
        # print('explanation.shape:',explanation.shape)
        prediction = predictor(graph_batch, explanation)
        # print('prediction.shape:',prediction.shape)

        train_right += torch.eq(prediction.argmax(dim = 1), graph_batch.y).sum()
        
        exp_pr, exp_re, exp_acc, dir_pr, roc_auc = exp_metric(explanation, graph_batch.edge_gt_att)
        train_exp_pr.append(exp_pr.cpu().item())
        train_exp_re.append(exp_re.cpu().item())
        train_exp_acc.append(exp_acc.cpu().item())
        train_dir_pr.append(dir_pr.cpu().item())
        train_roc_auc.append(roc_auc)
        
        exp_loss = Exp_loss(explanation, graph_batch.edge_gt_att.float())
        pred_loss = Pred_loss(prediction, graph_batch.y)
        total_loss = exp_loss + pred_loss
        
        train_exp_loss.append(exp_loss.cpu().item())
        train_pred_loss.append(pred_loss.cpu().item())
        train_total_loss.append(total_loss.cpu().item())
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()

    train_acc = train_right / len(train_loader.dataset)
    print('----------------- Training Epoch %d -----------------------' % epoch)
    print('Training predict accuracy:', train_acc.cpu().item())
    print('Training dir precision:', np.mean(train_dir_pr))
    print('Training roc auc:', np.mean(train_roc_auc))
    print('Training explain accuracy:', np.mean(train_exp_acc))    
    print('Training explain precision:', np.mean(train_exp_pr))
    print('Training explain recall:', np.mean(train_exp_re))
    print('Training explain loss:', np.mean(train_exp_loss),
          '\nTraining predict loss:', np.mean(train_pred_loss),
          '\nTraining total loss:', np.mean(train_total_loss))
    print('-------------------------------------------------------\n')
        
    if epoch % 2 == 0:
        val_right = 0
        val_exp_pr = []
        val_exp_re = []
        val_exp_acc = []
        val_dir_pr = []
        val_roc_auc = []
        val_exp_loss = []
        val_pred_loss=  []
        val_total_loss = []
        with torch.no_grad():
            for graph_batch in val_loader:
                graph_batch.to(device)

                graph_batch.x = F.relu(projector(graph_batch.x))
                explanation = explainer(graph_batch, 0, False)
                prediction = predictor(graph_batch, explanation)
                    
                val_right += torch.eq(prediction.argmax(dim = 1), graph_batch.y).sum()
                exp_pr, exp_re, exp_acc, dir_pr, roc_auc = exp_metric(explanation, graph_batch.edge_gt_att)
                val_exp_pr.append(exp_pr.cpu().item())
                val_exp_re.append(exp_re.cpu().item())
                val_exp_acc.append(exp_acc.cpu().item())
                val_dir_pr.append(dir_pr.cpu().item())
                val_roc_auc.append(roc_auc)
                    
                exp_loss = Exp_loss(explanation, graph_batch.edge_gt_att.float())
                pred_loss = Pred_loss(prediction, graph_batch.y)
                total_loss = exp_loss + pred_loss
                    
                val_exp_loss.append(exp_loss.item())
                val_pred_loss.append(pred_loss.item())
                val_total_loss.append(total_loss.item())
                    
            if max_val_loss > np.mean(val_total_loss):
                max_val_loss = np.mean(val_total_loss)
                torch.save(explainer.cpu(), osp.join('finetuned/Mnist/Explainer.pt'))
                torch.save(projector.cpu(), osp.join('finetuned/Mnist/Projector.pt'))
                torch.save(predictor.cpu(), osp.join('finetuned/Mnist/Predictor.pt'))
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
            print('Validation accuray:', val_acc.cpu().item())
            print('Validation dir precision:', np.mean(val_dir_pr))
            print('Validation roc auc:', np.mean(val_roc_auc))
            print('Validation explain accuracy:', np.mean(val_exp_acc))    
            print('Validation explain precision:', np.mean(val_exp_pr))
            print('Validation explain recall:', np.mean(val_exp_re))
            print('Validation explain loss:', np.mean(val_exp_loss),
                  '\nValidation predict loss:', np.mean(val_pred_loss),
                  '\nValidation total loss:', np.mean(val_total_loss))
            print('-------------------------------------------------------\n')

# In[Test]
best_explainer = torch.load('finetuned/Mnist/Explainer.pt').to(device)
best_projector = torch.load('finetuned/Mnist/Projector.pt').to(device)
best_predictor = torch.load('finetuned/Mnist/Predictor.pt').to(device)
test_right = 0
test_exp_pr = []
test_exp_re = []
test_exp_acc = []
test_dir_pr = []
test_roc_auc = []
test_exp_loss = []
test_pred_loss=  []
test_total_loss = []
for graph_batch in test_loader:
    graph_batch.to(device)

    graph_batch.x = F.relu(best_projector(graph_batch.x))
    explanation = best_explainer(graph_batch, 0, False)
    prediction = best_predictor(graph_batch, explanation)
    
    test_right += torch.eq(prediction.argmax(dim = 1), graph_batch.y).sum()
    exp_pr, exp_re, exp_acc, dir_pr, roc_auc = exp_metric(explanation, graph_batch.edge_gt_att)
    test_exp_pr.append(exp_pr.cpu().item())
    test_exp_re.append(exp_re.cpu().item())
    test_exp_acc.append(exp_acc.cpu().item())
    test_dir_pr.append(dir_pr.cpu().item())
    test_roc_auc.append(roc_auc)
    
    exp_loss = Exp_loss(explanation, graph_batch.edge_gt_att.float())
    pred_loss = Pred_loss(prediction, graph_batch.y)
    total_loss = exp_loss + pred_loss
    
    test_exp_loss.append(exp_loss.item())
    test_pred_loss.append(pred_loss.item())
    test_total_loss.append(total_loss.item())

test_acc = test_right / len(test_loader.dataset)
print('----------------- Testing -----------------------')
print('Testing predict accuracy:', test_acc.cpu().item())
print('Testing dir precision:', np.mean(test_dir_pr))
print('Testing roc auc:', np.mean(test_roc_auc))
print('Testing explain accuracy:', np.mean(test_exp_acc))    
print('Testing explain precision:', np.mean(test_exp_pr))
print('Testing explain recall:', np.mean(test_exp_re))
print('Testing explain loss:', np.mean(test_exp_loss),
      '\nTesting predict loss:', np.mean(test_pred_loss),
      '\nTesting total loss:', np.mean(test_total_loss))
print('-------------------------------------------------------\n')