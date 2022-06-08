import os, sys
#print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.parameters import Parameters
from src.core.moledataloader import MoleDataLoader
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', required=True)
parser.add_argument('--get_vocab', action="store_true")
args = parser.parse_args()

# load parameters and datasets

params = Parameters()
params.load(args.config)
dataloader = MoleDataLoader(params)

print(args.config, params)

###
# Option: Derive vocabulary from train set
#         Extract substructure vocabulary from a given set of molecules
###
if args.get_vocab:
    dataloader.preprocess_vocab(input_file=params.train_set, output_file=params.vocab_set)
    if params.valid_set != "":
        dataloader.preprocess_vocab(input_file=params.valid_set, output_file=params.vocab_set)  
    if params.ymols_set != "":
        dataloader.preprocess_vocab(input_file=params.ymols_set, output_file=params.vocab_set)  
    #dataloader.preprocess_vocab(input_file='benchmarks/logp04/mols.txt', output_file=params.vocab_set)
    #dataloader.preprocess_vocab(input_file='data/moses/vocab_train.txt', output_file=params.vocab_set)

dataloader.preprocess_data_tensor(params.train_set, params.train_mode, params.preprocess_save_dir)
if params.ymols_set != "":
    dataloader.preprocess_data_tensor(params.ymols_set, params.ymols_mode, params.ymols_preprocess_save_dir)  
#dataloader.load_preprocessed_train_set()
