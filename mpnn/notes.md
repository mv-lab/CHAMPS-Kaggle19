## notes for the mpnn

- features converted to graph pickle files in [data.py](data.py)

- graph files are read in the __getitem__ function in the [dataset.py](dataset.py) ChampsDataset class

- ChampsDataset class is used in [train.py](train.py), run_train to extract pickle files 

- null_collate is used when creating the ChampsDataset dataloader to combine the variable size graph features

- these features are passed to the [model.py](model.py) forward function for the mpnn model with lstm mechanism

## TODO

- more features (look at giba features)

- find better angle features 
    - right now it is:
    - `norm_xyz = preprocessing.normalize(xyz, norm='l2')`
    - `angle[ij] = (norm_xyz[i]*norm_xyz[j]).sum()` | does this make sense?
    - there are calculations for cosine and dehidral seperatly
    - cosine would benefit 2J couplings and dehidral 3J

- try training per coupling type / groups ['1JHC', '2JHC', '3JHC'] (in progress)

- try out sagpooling instead of set2set (in progress)

- try different structure from pytorch geometric (GraphConv?)