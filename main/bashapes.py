# In[Import]
import os
import torch
import numpy as np
import os.path as osp
from torch.nn import CrossEntropyLoss, BCELoss

import model
from utils import utils4nc
from utils import eval4nc
from utils import mask
from dataset.dataset4nc import load_dataset
from dataset.dataset4nc import load_dataset_ground_truth


# In[Device]
cuda = 3
if torch.cuda.is_available():
    device = torch.device('cuda:%d' % cuda)
else:
    device = torch.device('cpu')

# In[Save path]
save_path = '/home/data/yj/Bnova/trained/BAShapes'
os.makedirs(save_path, exist_ok = True)

# In[Dataset]
_dataset = 'BAShapes'
graph, features, node_labels, train_mask, val_mask, test_mask = load_dataset(_dataset)
exp, indices = load_dataset_ground_truth(_dataset)
indices = torch.tensor(np.argwhere(test_mask).squeeze()).to(device)

# torch.Size([700, 10])
x = torch.tensor(features).to(device)
# torch.Size([2, 4110])
edge_index = torch.tensor(graph).to(device)
# torch.Size([700])
node_labels = torch.tensor(node_labels).to(device)
# torch.Size([4110])
exp_labels = torch.tensor(exp[1]).to(device)

num_nodes, node_in_dim = x.shape[0], x.shape[1]
classes = 4

# In[Initialize]
svd_out_dim = 256
svd_seed = list(np.random.randint(500, size = 8))
mcembedding = model.MCEmbedding(svd_out_dim, svd_seed).to(device)
diffusion = model.Diffusion(svd_out_dim, ratio = 0.25, decay = 0.999).to(device)

e2r_hidden_dim = 128
e2r_out_dim = 64
e2rfunction = model.E2RFuncttion(svd_out_dim, e2r_hidden_dim, e2r_out_dim).to(device)

hyper_hidden_dim = 128
hyperweight = model.HyperWeight(e2r_out_dim, hyper_hidden_dim, node_in_dim).to(device)
consensus = model.Consensus().to(device)

predictor = model.NodePredictor(node_in_dim, classes).to(device)

exp_evaluator = eval4nc.AUCEvaluation((edge_index,exp_labels))

# In[Loss]
Pred_loss = CrossEntropyLoss()
Exp_loss = BCELoss()

# In[Optimizer]
exp_lr = 1e-2
pred_lr = 1e-1
params_dict = [{'params': e2rfunction.parameters(), 'lr': exp_lr},
               {'params': hyperweight.parameters(), 'lr': exp_lr},
               {'params': predictor.parameters(), 'lr': pred_lr}]
opt = torch.optim.Adam(params_dict)

# In[Train, Valid and Save]
epochs = 20
best_val_acc = 0.0
best_val_auc = 0.0

for epoch in range(1, epochs + 1):
    opt.zero_grad()
    mc_embeddings = mcembedding(edge_index)
    mc_embeddings_diffused = diffusion(mc_embeddings, epoch)
    edge_reps = e2rfunction(edge_index, mc_embeddings_diffused)
    hyper_weights = hyperweight(edge_index, edge_reps, num_nodes, x)
    weights = consensus(hyper_weights).squeeze()

    exp_sparsity = utils4nc.sparsity(weights)
    val_auc = eval4nc.roc_auc(weights.detach().cpu().numpy(),
                              exp_labels.detach().cpu().numpy())
    exp_loss = Exp_loss(weights, exp_labels.float())

    mask.set_mask(weights, predictor)
    out = predictor(x, edge_index, weights)
    pred_loss = Pred_loss(out[train_mask], node_labels[train_mask])
    mask.clear_mask(predictor)

    total_loss = exp_loss + pred_loss
    total_loss.backward()
    opt.step()

    train_acc = eval4nc.evaluate(out[train_mask], node_labels[train_mask])
    val_acc = eval4nc.evaluate(out[val_mask], node_labels[val_mask])
    test_acc = eval4nc.evaluate(out[test_mask], node_labels[test_mask])

    if val_acc > best_val_acc:
        print('Validate prediction improved.')
        best_val_acc = val_acc
        torch.save(predictor.cpu(),osp.join('trained/BAShapes/Predictor.pt'))
        predictor.to(device)
    if val_auc > best_val_auc:
        print('Validate explanation improved.')
        best_val_auc = val_auc
        torch.save(e2rfunction.cpu(), osp.join('trained/BAShapes/E2RFunction.pt'))
        torch.save(hyperweight.cpu(), osp.join('trained/BAShapes/Weightor.pt'))
        e2rfunction.to(device)
        hyperweight.to(device)

    print('----------------- Epoch %d -----------------' % epoch)
    print(f'train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, test_acc:{test_acc:.4f}')
    print(f'explanation auc: {val_auc:.4f}')
    print(f'train_exp_loss: {exp_loss:.4f}, train_pred_loss: {pred_loss:.4f},\
          train_total_loss: {total_loss:.4f}')
    print('--------------------------------------------\n')

# In[Test]
best_e2rfunction = torch.load('trained/BAShapes/E2RFunction.pt').to(device)
best_weightor = torch.load('trained/BAShapes/Weightor.pt').to(device)
best_predictor = torch.load('trained/BAShapes/Predictor.pt').to(device)

mc_embeddings = mcembedding(edge_index)
mc_embeddings_diffused = diffusion(mc_embeddings, epoch)
edge_reps = best_e2rfunction(edge_index, mc_embeddings_diffused)
hyper_weights = best_weightor(edge_index, edge_reps, num_nodes, x)
weights = consensus(hyper_weights)

exp_sparsity = utils4nc.sparsity(weights)
test_auc = eval4nc.roc_auc(weights.detach().cpu().numpy(),
                           exp_labels.detach().cpu().numpy())
exp_loss = Exp_loss(weights, exp_labels)

mask.set_mask(weights, best_predictor)
out = best_predictor(x, edge_index, weights)
mask.clear_mask(best_predictor)

train_acc = eval4nc.evaluate(out[train_mask], node_labels[train_mask])
val_acc = eval4nc.evaluate(out[val_mask], node_labels[val_mask])
test_acc = eval4nc.evaluate(out[test_mask], node_labels[test_mask])

print('----------------- Test -----------------')
print(f'train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, test_acc:{test_acc:.4f}')
print(f'explanation auc: {test_auc:.4f}')
print('----------------------------------------\n')
