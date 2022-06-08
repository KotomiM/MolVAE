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
from tqdm import tqdm

import rdkit

from src.core.parameters import Parameters
from src.core.moledataloader import MoleDataLoader, MolEnumRootDataset, MolTreeDataset
from src.models.model_factory import ModelFactory
from src.models.fast_jtnn import MolTree
from src.common.vocab import PairVocab

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', required=True)
parser.add_argument('--model', type=str, default='', required=True)
args = parser.parse_args()

print('load config from: ', args.config)


#torch.manual_seed(args.seed)
#random.seed(args.seed)


# load config and data
params = Parameters()
params.load(args.config)

torch.manual_seed(params.seed)
random.seed(params.seed)
print(params)


moledataloader = MoleDataLoader(params)
moledataloader.load_vocab_set()


#moledataloader.load_valid_set()
# load model
params.vocab = moledataloader.vocab
params.atom_vocab = moledataloader.atom_vocab

model = ModelFactory.get_model(params)
model = model.cuda()
model.load_state_dict(torch.load(args.model))
results=""
if params.model == 'hiervae':
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(params.nsample // params.batch_size)):
            smiles_list = model.sample(params.batch_size, greedy=True)
            for _,smiles in enumerate(smiles_list):
                results+=smiles+'\n'
                print(smiles)

elif params.model == 'jtvae':
    for i in range(params.nsample):
        s=model.sample_prior()
        results+=s+'\n'
        print(s)
   
            
if not os.path.exists(params.decode_save_dir):
    os.makedirs(params.decode_save_dir)

with open(params.decode_save_dir+'/'+params.model+"_"+ params.rnn_type+"_results.csv", 'w') as f:
    f.write(results)
