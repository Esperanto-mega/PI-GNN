# In[Import]
import torch
import torch_geometric
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torch.nn import ReLU, Linear
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import HypergraphConv
from sklearn.decomposition import TruncatedSVD
from collections import OrderedDict
from utils import unbatch
from utils import mask

# In[Monte Carlo Embedding]
class MCEmbedding(torch.nn.Module):
    def __init__(self, out_dim, svd_seed = [0], svd_iter = 5):
        super().__init__()
        self.out_dim = out_dim
        self.svds = []
        for seed in svd_seed:
            self.svds.append(TruncatedSVD(out_dim, n_iter = svd_iter, 
                                          random_state = seed))
            
    def forward(self, edge_index):
        mc_embeddings = []
        device = edge_index.device
        adj = to_dense_adj(edge_index).cpu().squeeze()
        # print('adj.shape', adj.shape)
        if adj.shape[0] < self.out_dim:
            delta = self.out_dim - adj.shape[0] + 1
            adj = F.pad(adj, (0, delta, 0, delta))
        elif adj.shape[0] == self.out_dim:
            adj = F.pad(adj, (0, 1, 0, 1))
          
        for svd in self.svds:
            mc_embedding = torch.tensor(svd.fit_transform(adj), device = device)
            # print('mc_embedding.shape', mc_embedding.shape)
            mc_embeddings.append(mc_embedding)
            
        return mc_embeddings

# In[Variational Diffusion]
class Diffusion(torch.nn.Module):
    def __init__(self, in_dim, ratio, decay):
        super().__init__()
        self.in_dim = in_dim
        self.diffuse_ratio = ratio
        self.decay = decay
        
    def forward(self, mc_embeddings, t):
        if t == 0:
            # print('mc_embeddings[0].device', mc_embeddings[0].device)
            return mc_embeddings

        damping = np.power(self.decay, t)
        num_nodes = mc_embeddings[0].shape[0]
        mc_tensor_view = torch.cat(mc_embeddings).reshape(-1, num_nodes, self.in_dim)
        assert mc_embeddings[0].equal(mc_tensor_view[0])
        # print('mc_tensor_view.shape', mc_tensor_view.shape)
        mean = torch.mean(mc_tensor_view, dim = 0) * np.sqrt(damping)
        std = torch.std(mc_tensor_view, dim = 0) * np.sqrt(1 - damping)
        # print('mean.shape', mean.shape)
        # print('std.shape', std.shape)
        
        for mc_embedding in mc_embeddings:
            gaussian = torch.randn(num_nodes, self.in_dim, device = mc_embedding.device)
            sampling = self.diffuse_ratio * (mean + std * gaussian)
            mc_embedding = (1 - self.diffuse_ratio) * mc_embedding + sampling
            # print('mc_embedding_diffused.shape', mc_embedding.shape)
            
        return mc_embeddings
       
# In[Transform Function]
class E2RFuncttion(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.transformation = nn.Sequential(OrderedDict([
            ('Linear1', nn.Linear(2 * in_dim, hidden_dim)),
            ('Activate', nn.ReLU()),
            ('Linear2', nn.Linear(hidden_dim, out_dim))]))
        
        self.weight_init_(mode = 'kaiming')
        
    def forward(self, edge_index, mc_embeddings):
        edge_reps = []
        for mc_embedding in mc_embeddings:
            edge_rep = torch.cat([mc_embedding[edge_index[0,:]],
                                  mc_embedding[edge_index[1,:]]], dim = 1)
            # print('edge_rep.shape', edge_rep.shape)
            # print('edge_rep.device', edge_rep.device)
            edge_reps.append(self.transformation(edge_rep))
        
        return edge_reps
    
    def weight_init_(self,mode = 'kaiming'):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if mode == 'kaiming':
                    nn.init.kaiming_normal_(module.weight)
                elif mode == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.normal_(module.weight)

# In[Hypergraph]
class HyperWeight(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, node_in_dim = 0, out_dim = 1):
        super().__init__()
        self.in_dim = in_dim
        self.node_in_dim = node_in_dim
        if node_in_dim != 0:
            self.in_dim += node_in_dim
        # print('self.in_dim', self.in_dim)
        
        self.hidden_dim = hidden_dim
        
        self.hyperconv1 = HypergraphConv(self.in_dim, hidden_dim)
        self.hyperconv2 = HypergraphConv(hidden_dim, out_dim)
        
    def forward(self, edge_index, edge_reps, num_nodes, 
                node_feature, edge_feature = None):
        hyperedge_index = self.edge2hyperedge(edge_index, num_nodes)
        hyperedge_index.to(edge_index.device)
        # print('hyperedge_index.device', hyperedge_index.device)
        
        edge_features = []
        if node_feature != None:
            for edge in edge_index.T:
                f_u, f_v = node_feature[edge[0]], node_feature[edge[1]]
                # print('f_u.shape', f_u.shape)
                # print('f_v.shape', f_v.shape)
                edge_features.append((f_u + f_v) / 2)
                
        edge_features = torch.cat(edge_features).reshape(-1, self.node_in_dim)
        # print('edge_features.shape', edge_features.shape)
        
        hyper_weights = []
        for edge_rep in edge_reps:
            in_rep = torch.cat([edge_rep, edge_features], dim = 1)
            # print('in_rep.shape', in_rep.shape)
            # print('in_rep.device', in_rep.device)
            edge_rep = self.hyperconv1(in_rep, hyperedge_index)
            edge_rep = torch.tanh(edge_rep)
            edge_rep = self.hyperconv2(edge_rep, hyperedge_index)
            hyper_weight = torch.sigmoid(edge_rep)
            # print('hyper_weight.shape', hyper_weight.shape)
            hyper_weights.append(hyper_weight)
        
        return hyper_weights
    
    def edge2hyperedge(self, edge_index, num_nodes):
        hyperedges = [[] for i in range(num_nodes)]
        num_edges = edge_index.shape[1]
        for i in range(num_edges):
            u, v = edge_index.T[i]
            hyperedges[u].append(i)
            hyperedges[v].append(i)
        
        hyperedge_index = []
        for i in range(num_nodes):
            hyperedge = torch.tensor(hyperedges[i], dtype = torch.long,
                                     device = edge_index.device)
            index = torch.empty_like(hyperedge).fill_(i).to(edge_index.device)
            hyperedge_index.append(torch.stack([hyperedge, index]))
        return torch.cat(hyperedge_index, dim = 1)

# In[Consensus]
class Consensus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, hyper_probs):
        # print('hyper_prob.shape', hyper_probs[0].shape)
        # print('len(hyper_probs)',len(hyper_probs))
        if len(hyper_probs) == 1:
            return hyper_probs[0]
        
        '''multiple ensemble'''
        # consensus_prob = torch.ones_like(hyper_probs[0])
        # for hyper_prob in hyper_probs:
        #     # print('hyper_prob.shape', hyper_prob.shape)
        #     consensus_prob *= hyper_prob

        '''weighted sum ensemble'''
        # alpha = []
        # mean_prob = torch.mean(torch.stack(hyper_probs), dim = 0)
        # # print('mean_prob.shape',mean_prob.shape)
        # for hyper_prob in hyper_probs:
        #     # print('mean_prob.T.shape',mean_prob.T.shape)
        #     alpha.append(torch.mm(mean_prob.T,hyper_prob).item())
        # normal_alpha = [x/sum(alpha) for x in alpha]
        # consensus_prob = torch.zeros_like(hyper_probs[0])
        # for i in range(len(hyper_probs)):
        #     consensus_prob += normal_alpha[i] * hyper_prob[i]

        '''mean ensemble'''
        consensus_prob = torch.mean(torch.stack(hyper_probs), dim = 0)

        return consensus_prob

class SVDExplainer(torch.nn.Module):
    def __init__(self, node_in_dim,
                 svd_dim, e2r_hidden_dim, e2r_out_dim, hyper_dim,
                 svd_seed = [0], svd_iter = 5,
                 dif_ratio = 0.25, dif_decay = 0.999):
        super().__init__()
        
        self.node_in_dim = node_in_dim
        self.svd_dim = svd_dim
        self.e2r_hidden_dim = e2r_hidden_dim
        self.e2r_out_dim = e2r_out_dim
        self.hyper_dim = hyper_dim
        self.svd_seed = svd_seed
        self.svd_iter = svd_iter
        self.dif_ratio = dif_ratio
        self.dif_decay = dif_decay

        self.MCEmbedding = MCEmbedding(svd_dim, svd_seed, svd_iter)
        self.Diffusion = Diffusion(svd_dim, dif_ratio, dif_decay)
        self.E2RFuncttion = E2RFuncttion(svd_dim, e2r_hidden_dim, e2r_out_dim)
        self.HyperWeight = HyperWeight(e2r_out_dim, hyper_dim, node_in_dim)
        self.Consensus = Consensus()

    def forward(self, g, t, need_edge_pool = True):
        num_nodes = g.x.shape[0]
        # print('g.x.device', g.x.device)
        
        unbatch_edge_index = unbatch.unbatch_edge_index(g.edge_index.long(), 
                                                        g.batch.long())
        
        weights = []
        edge_pool = []
        for ei in unbatch_edge_index:
            # print('ei.device', ei.device)
            mc_embeddings = self.MCEmbedding(ei)
            mc_embeddings_diffused = self.Diffusion(mc_embeddings, t)
            edge_reps = self.E2RFuncttion(ei, mc_embeddings_diffused)
            tensor_view = torch.cat(edge_reps)
            # print('tensor_view.shape', tensor_view.shape)
            edge_pool.append(torch.mean(tensor_view, dim = 0, keepdim = True))
            hyper_weights = self.HyperWeight(ei, edge_reps, num_nodes, g.x)
            weights.append(self.Consensus(hyper_weights))
            
        weights = torch.cat(weights, dim = 0).squeeze()

        if need_edge_pool is True:
            edge_pool = torch.cat(edge_pool, dim = 0)
            return weights, edge_pool
        else:
            return weights

class GraphPredictor(torch.nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim,
                 edge_dim, graph_class = 2):
        super().__init__()

        self.Encoder = GCNConv(node_in_dim, node_hidden_dim)
        self.Predictor = nn.Linear(node_hidden_dim + edge_dim, graph_class)

    def forward(self, g, weights, edge_pool = None):
        mask.set_mask(weights, self.Encoder)
        structural_rep = F.relu(self.Encoder(g.x, g.edge_index.long(), weights))
        mask.clear_mask(self.Encoder)

        graph_rep = global_mean_pool(structural_rep, g.batch)
        if edge_pool is not None:
            graph_rep = torch.cat([graph_rep, edge_pool], dim = 1)
        
        '''Unnormalized logits'''
        logits = self.Predictor(graph_rep)
        return logits

class NodePredictor(torch.nn.Module):
    """
    A node clasification model for nodes decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.embedding_size = 32 * 3
        self.conv1 = GCNConv(num_features, 32)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(32, 32)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(32, 32)
        self.relu3 = ReLU()
        self.lin = Linear(32 * 3, num_classes)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        final = self.lin(input_lin)
        return final

    def embedding(self, x, edge_index, edge_weights=None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p=2, dim=1)  # this is not used in PGExplainer
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p=2, dim=1)  # this is not used in PGExplainer
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p=2, dim=1)  # this is not used in PGExplainer
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin

class MainModel(torch.nn.Module):
    def __init__(self, node_in_dim, node_hidden_dim, 
                 svd_dim, e2r_hidden_dim, e2r_out_dim, hyper_dim,
                 svd_seed = [0], svd_iter = 5, 
                 dif_ratio = 0.25, dif_decay = 0.999,
                 node_class = None, graph_class = None, task = 'node'):
        super().__init__()
        
        self.Explainer = SVDExplainer(node_in_dim,
                                      svd_dim, e2r_hidden_dim, e2r_out_dim, 
                                      hyper_dim,
                                      svd_seed, svd_iter, 
                                      dif_ratio, dif_decay)

        self.e2r_out_dim = e2r_out_dim

        if task == 'node':
            self.Predictor = NodePredictor(node_in_dim, node_hidden_dim, node_class)
        elif task == 'graph':
            self.Predictor = GraphPredictor(node_in_dim, node_hidden_dim, 
                                            e2r_out_dim, graph_class)
        elif task == 'link':
            pass

        self.task = task

    def forward(self, g, t, explain = True):
        weights, edge_pool = self.Explainer(g, t)
        self.edge_pool = edge_pool
        if self.task == 'graph':
            logits = self.Predictor(g, weights, edge_pool)
        elif self.task == 'node':
            logits = self.Predictor(g, weights)

        if explain is True:
            return weights, logits
        else:
            return logits
        
    def explain_forward(self, g, mask):
        edge_pool = torch.randn_like(self.edge_pool)
        if self.task == 'graph':
            logits = self.Predictor(g, mask, edge_pool)
            return logits
        elif self.task == 'node':
            logits = self.Predictor(g, mask)
            return logits