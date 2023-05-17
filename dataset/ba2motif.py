# In[Import]
import random
import numpy as np
import os.path as op

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# In[BA2Motif]
class BA2Motif(InMemoryDataset):
    splits = ['train', 'valid', 'test']
    
    def __init__(self, root, mode, 
                 transform = None,
                 pre_transform = None,
                 pre_filter = None):
        assert mode in self.splits
        self.mode = mode
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
        idx = self.processed_file_names.index('{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])
        
    @property
    def raw_file_names(self):
        return 'ba2motif.pkl'
    
    
    @property
    def processed_file_names(self):
        return ['train.pt','valid.pt','test.pt']
    
    def download(self):
        file = 'ba2motif.pkl'
        # print('self.raw_dir:', self.raw_dir)
        # print('op.join(self.raw_dir, file):',op.join(self.raw_dir, file))
        if not op.exists(op.join(self.raw_dir, file)):
            print('Data does not exist.')
            raise FileNotFoundError
            
    def process(self):
        # print('op.join(self.raw_dir,self.raw_file_names):',op.join(self.raw_dir,self.raw_file_names))
        adj, x, y, exp_gt = np.load(op.join(self.raw_dir,
                                            self.raw_file_names),
                                    allow_pickle = True)
        
        graph_list = []
        
        for i, (adj, x, y, exp_gt) in enumerate(zip(adj, x, y, exp_gt)):
            edge_index = dense_to_sparse(torch.tensor(adj))[0]
            node_index = torch.unique(edge_index)
            assert node_index.max() == node_index.size(0) - 1
            
            x = torch.tensor(x, dtype = torch.float)
            y = np.argmax(y)
            y = torch.tensor(y, dtype = torch.long)
            exp_gt = torch.tensor(exp_gt, dtype = torch.long)
            
            graph = Data(x = x,
                         edge_index = edge_index,
                         y = y,
                         exp_gt = exp_gt)
            
            if self.pre_filter is not None:
                graph = self.pre_filter(graph)
            
            if self.pre_transform is not None:
                graph = self.pre_transform(graph)
            
            graph_list.append(graph)
            
        random.shuffle(graph_list)
        
        torch.save(self.collate(graph_list[0:400]),
                   self.processed_paths[0])
        torch.save(self.collate(graph_list[400:800]),
                   self.processed_paths[1])
        torch.save(self.collate(graph_list[800:]),
                   self.processed_paths[2])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    