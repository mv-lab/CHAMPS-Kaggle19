import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob

def calculate_padding(matrix_size, pad_size):
    total_padding = pad_size - matrix_size

    if total_padding % 2 == 0:
        before = after = total_padding / 2
    else:
        before = (total_padding / 2) + 0.5
        after = total_padding - before

    return int(before), int(after)
    
    
class MolData(Dataset):
    def __init__(self, fnames, inpath, graph_path, node_path, angle_path, outpath=None, 
                 pad_size=29, norms=None, angle_norms=None, coupling_types=None):
        super(MolData, self).__init__()
        self.fnames = np.array(fnames)
        self.inpath = inpath
        self.graph_path = graph_path
        self.node_path = node_path
        self.angle_path = angle_path
        self.outpath = outpath
        
        self.pad_size = pad_size
        self.norms = norms
        self.angle_norms = angle_norms
        self.coupling_types = coupling_types
    
        
    def __len__(self):
        return len(self.fnames)
        
    def __getitem__(self, idx):
        #first load the numpy array data
        fname = self.fnames[idx]
        in_file = self.inpath + '/' + fname
        matrices = np.load(in_file)
        graph = np.load(self.graph_path + '/' + fname)
        nodes = np.load(self.node_path + '/' + fname)
        angles = np.load(self.angle_path + '/' + fname)
        
        #this is needed for self
        graph[np.where(graph == 0)] = 1
        
        if self.outpath:
            target_file = self.outpath + '/' + fname
            target = np.load(target_file)
        else:
            target = np.zeros(matrices.shape[1:])
            
        if self.norms is not None:
            matrices[0] = (matrices[0] - self.norms[0]) / self.norms[1]
            
        if self.angle_norms is not None:
            angles[0][graph == 2] = (angles[0][graph == 2] - self.angle_norms[0][0]) / self.angle_norms[0][1]
            angles[1][graph == 3] = (angles[1][graph == 3] - self.angle_norms[1][0]) / self.angle_norms[1][1]
            
        if self.pad_size is not None:    
            before, after = calculate_padding(target.shape[0], self.pad_size)
            matrices = np.pad(matrices, ((0, 0), (before, after), (before, after)), mode='constant')
            angles = np.pad(angles, ((0, 0), (before, after), (before, after)), mode='constant')
            graph = np.pad(graph, (before, after), mode='constant')
            nodes = np.pad(nodes, (before, after), mode='constant')
            target = np.pad(target, (before, after), mode='constant')    

        
        #the first layer is the distance matrix
        #second layer are the atom/bond types
        #third layer are the coupling types
        dist_mat, _, type_mat = matrices
        
                
        #if there are coupling types specified, adjust the
        #type_matrix to only contain them
        if self.coupling_types is not None:
            ni = 1
            new_type_mat = np.zeros_like(type_mat)
            for ct in self.coupling_types:
                new_type_mat[type_mat == ct] = ni
                ni += 1
                
            type_mat = np.copy(new_type_mat)
        
        #we only need to take the first row of the bond_mat
        #since the atom types are the same for each row
        
        
        #now we need to convert our data to torch tensors
        #dist_mat and target are converted to float
        #dist_mat needs to be normalized
        #bond_mat and type_mat need to be converted to long type
        dist_mat = torch.tensor(dist_mat).float()
        angle_mat = torch.tensor(angles).float()
        target = torch.tensor(target).float()
        #bond_mat = torch.tensor(bond_mat).long()
        nodes = torch.tensor(nodes).long()
        type_mat = torch.tensor(type_mat).long()
        graph = torch.tensor(graph).long()
        
        return {'fname': fname,
                'distance': dist_mat,
                'angle': angle_mat,
                'atoms': nodes,
                'type': type_mat,
                'graph': graph,
                'target': target
               }
