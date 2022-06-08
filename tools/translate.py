import os, sys
#print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse

import rdkit

from src.core.parameters import Parameters
from src.core.moledataloader import MoleDataLoader, MolEnumRootDataset, MolTreeDataset
from src.models.model_factory import ModelFactory
from src.models.fast_jtnn import MolTree

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', required=True)
parser.add_argument('--model', type=str, default='', required=True)
args = parser.parse_args()

print('load config from: ', args.config)

# load config and data
params = Parameters()
params.load(args.config)
print(params)
moledataloader = MoleDataLoader(params)
moledataloader.load_vocab_set()
moledataloader.load_test_set()
# moledataloader.load_valid_set()
# load model
params.vocab = moledataloader.vocab
params.atom_vocab = moledataloader.atom_vocab
params.test = moledataloader.test

model = ModelFactory.get_model(params)
model = model.cuda()
model.load_state_dict(torch.load(args.model))

if params.model == 'hiervae' or params.model == 'hiervgnn':
    model.eval()

    params.enum_root = True
    params.greedy = not params.sample
    """
    # choice for no vairational
    if params.novi:
        model = HierGNN(params).cuda()
    else:
        model = HierVGNN(params).cuda()
    """

    dataset = MolEnumRootDataset(params.test, params.vocab, params.atom_vocab)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

    torch.manual_seed(params.seed)
    random.seed(params.seed)

    results = ""
    with torch.no_grad():
        for i, batch in enumerate(loader):
            smiles = params.test[i]
            if batch is None:
                for k in range(params.num_decode):
                    results += smiles+' '+smiles+'\n'
                    #print(smiles, smiles)
            else:
                new_mols = model.translate(batch[1], params.num_decode, params.enum_root, params.greedy)
                for k in range(params.num_decode):
                    results += smiles+' '+new_mols[k]+'\n'
                    #print(smiles, new_mols[k]) 
            """
            if i > params.test_num:
                break
            """

elif params.model == 'vjtnn':
    data = [MolTree(s) for s in moledataloader.test]
    print('Test on %d moleculars' % len(data))
    batches = [data[i : i + 1] for i in range(0, len(data))]
    dataset = MolTreeDataset(batches, params.vocab, assm=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

    torch.manual_seed(params.seed)
    
    #i = 0
    results = ""
    with torch.no_grad():
        for batch in loader:
            mol_batch = batch[0]
            x_tree_vecs, _, x_mol_vecs = model.encode(batch[1], batch[2])
            assert x_tree_vecs.size(0) == x_mol_vecs.size(0)
            
            for k in range(params.num_decode):
                z_tree_vecs, z_mol_vecs = model.fuse_noise(x_tree_vecs, x_mol_vecs)
                smiles = mol_batch[0].smiles
                #print('decoding')
                new_smiles = model.decode(z_tree_vecs[0].unsqueeze(0), z_mol_vecs[0].unsqueeze(0))
                if new_smiles == None:
                    new_smiles = "None"
                results += smiles+' '+new_smiles+'\n'
                #print(smiles, new_smiles)            
            """
            i += 1
            print(i)
            if i > params.test_num:
                break
            """
if not os.path.exists(params.decode_save_dir):
    os.makedirs(params.decode_save_dir)

suffix = args.model.split('.')[1]

with open(params.decode_save_dir+'/'+params.model+"_"+ params.rnn_type+'_'+suffix+"_results.csv", 'w') as f:
    f.write(results)
