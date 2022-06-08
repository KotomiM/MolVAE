import sys, os
import gc
import time
import torch
import random
import pickle
from functools import partial
from multiprocessing import Pool
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from src.common.vocab import PairVocab, Vocab, common_atom_vocab
from src.models.hgraph import MolGraph
from src.models.fast_jtnn import MolTree
from src.utils.datautils import DataFolder, MolEnumRootDataset, MolTreeDataset, PairTreeDataset, MolTreeFolder, PairTreeFolder
from src.utils.datautils import moltree_tensorize, jtvae_tensorize, vjtnn_tensorize, vjtnn_tensorize_pair, hgraph_tensorize, hgraph_tensorize_pair, hgraph_tensorize_cond

class MoleDataLoader(object):
    """
    @brief molecular training set and vocabulary dataloader for model training batch
    """
    def __init__(self, params):
        """
        @brief initialization
        """
        self.model = None       # model config in ['jtnn', 'graph2graph', 'hiervae', 'hiervgnn']
        
        self.data = None        # train set
        self.test = None        # test set
        self.valid = None       # valid set
        self.vocab = None       # vocabulary list
        self.atom_vocab = None  # atom vocab choice
        self.train_set = None   # dataset file path for model training
        self.train_mode = None  # training set mode in ['single', 'pair', 'cond_pair']

        self.vocab_set = None   # vocabulary file path for model training
        self.vocab_mode = None  # vocabulary mode in ['Vocab', 'PairVocab']
        self.valid_set = None   # validation set, mode = 'single'
        self.target_set = None  # target set, mode = 'single'
        self.test_set = None    # test set, mode = 'single'

        self.preprocess_save_dir = None # preprocessed data
        self.dataset = None             # DataFolder for training
        self.ncpu = None        # for preprocessing
        self.batch_size = None  # batch size

        self.model = params.model
        self.atom_vocab = common_atom_vocab
        self.train_set = params.train_set
        self.train_mode = params.train_mode
        self.vocab_set = params.vocab_set
        self.vocab_mode = params.vocab_mode

        self.test_set = params.test_set
        self.valid_set = params.valid_set
        self.preprocess_save_dir = params.preprocess_save_dir
        self.ncpu = params.ncpu
        self.batch_size = params.batch_size
        

    def load_preprocessed_train_set(self):
        """
        @brief load preprocessed train set
        """          
        # tensorized training sets file
        self.dataset = self.get_dataset()

    def get_dataset(self):
        """
        @brief method: return a dataset
        """         
        print("load dataset from preprocessed folder: ", self.preprocess_save_dir)
        if self.model == 'hiervae' or self.model == 'hiervgnn':
            return DataFolder(self.preprocess_save_dir, self.batch_size)
        if self.model == 'vjtnn':
            return PairTreeFolder(self.preprocess_save_dir, self.vocab, self.batch_size)
        if self.model == 'jtvae':
            return MolTreeFolder(self.preprocess_save_dir, self.vocab, self.batch_size, assm=True, num_workers=4)
        
    def load_vocab_set(self):
        """
        @brief load preprocessed vocab set
        """             
        # vocabulary initialization   
        if self.vocab_mode == 'Vocab':
            vocab = [x.strip("\r\n ") for x in open(self.vocab_set)] 
            self.vocab = Vocab(vocab)
        elif self.vocab_mode == 'PairVocab':
            vocab = [x.strip("\r\n ").split() for x in open(self.vocab_set)] 
            self.vocab = PairVocab(vocab)
        else:
            print('MoleDataLoader Error: Not recognized vocab_mode (%s), please input in ["Vocab", "PairVocab"]' % self.vocab_mode)
            exit()
    
    def load_valid_set(self):
        """
        @brief load validation set
        """             
        print("load validation set from: ", self.valid_set)
        self.valid = [line.strip("\r\n ") for line in open(self.valid_set)]

    def load_test_set(self):
        """
        @brief load test set
        """             
        print("load test set from: ", self.test_set)
        self.test = [line.strip("\r\n ") for line in open(self.test_set)]

    def preprocess_vocab(self, input_file, output_file):
        """
        @brief Extract vocab from input molecular smiles file to output file, it may take a long time without multiprocessing.
        """  
        print("Extract vocab from input molecular smiles file: ", input_file, " save append to : ", output_file)
        vocab_result = ''
        if self.model == 'jtvae' or self.model == 'vjtnn':
            cset = set()
            with open(input_file, 'r') as f:
                for line in f.readlines():
                    smiles_list = line.split()
                    for smiles in smiles_list:
                        mol = MolTree(smiles)
                        for c in mol.nodes:
                            cset.add(c.smiles)
            for x in cset:
                vocab_result = vocab_result + str(x) + '\n'
        elif self.model == 'hiervae' or self.model == 'hiervgnn':
            cset = set()
            with open(input_file, 'r') as f:
                for line in f.readlines():
                    smiles_list = line.strip("\r\n ").split()[:2]
                    for s in smiles_list:
                        hmol = MolGraph(s)
                        for node,attr in hmol.mol_tree.nodes(data=True):
                            smiles = attr['smiles']
                            cset.add( attr['label'] )
                            for i,s in attr['inter_label']:
                                cset.add( (smiles, s) )
            for item in cset:
                x, y = item
                vocab_result = vocab_result + str(x) + ' ' + str(y) + '\n'
            
        with open(output_file, 'a') as f:
            f.write(vocab_result)
        # return res
                    
    
    def preprocess_data_tensor(self, filename, mode, save_dir):
        """
        @brief read data and preprocess for model
        """             
        tt = time.time()
        random.seed(1)
        # Load preprocessed vocab set
        self.load_vocab_set()
        # (Loading) training set initialization
        if mode == 'single':
            #dataset contains single molecules
            with open(filename) as f:
                data = [line.strip("\r\n ").split()[0] for line in f]
        elif mode == 'pair':
            #dataset contains molecule pairs
            with open(filename) as f:
                data = [line.strip("\r\n ").split()[:2] for line in f]
        elif mode == 'cond_pair':
            #dataset contains molecule pairs with conditions
            with open(filename) as f:
                data = [line.strip("\r\n ").split()[:3] for line in f]
        else:
            print('MoleDataLoader Error: Unrecognized mode (%s), please input in ["single", "pair", "cond_pair"]' % self.vocab_mode)
            exit()

        random.shuffle(data)
        print("loading dataset takes %.3f seconds" % (time.time() - tt))

        # preprocess data for particular model
        tt = time.time()
        print('save preprocessed data in directory : ', save_dir)
        if os.path.exists(save_dir):
            print('MoleDataLoader Error: Folder %s exists, please remove and retry' % save_dir)
            exit()
        os.makedirs(save_dir)                

        batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]

        func = None
        all_data = []
        pool = Pool(self.ncpu)
        
        if self.model == 'jtvae':
            func = partial(jtvae_tensorize, assm=True)
            all_data = pool.map(func, data)
        elif self.model == 'vjtnn':
            if mode == 'pair':
                func = partial(vjtnn_tensorize_pair)
                all_data = pool.map(func, data)
            # preprocess target.txt
            elif mode == 'single':
                func = partial(vjtnn_tensorize, assm=False)
                all_data = pool.map(func, data)
        elif self.model == 'hiervae' or self.model == 'hiervgnn':
            if mode == 'pair':
                func = partial(hgraph_tensorize_pair, vocab=self.vocab)
            elif mode == 'cond_pair':
                func = partial(hgraph_tensorize_cond, vocab=self.vocab)
            elif mode == 'single':
                func = partial(hgraph_tensorize, vocab=self.vocab)
            all_data = pool.map(func, batches)
        
        #all_data = [func(batch) for batch in batches]
        # Equals to: all_data = [func(batch) for batch in batches]
        
        num_splits = max(len(all_data) // 1000, 1)
        # Hint: it takes a long time around 40 minutes ~ several hours
        print('all_data prepared')
        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open(os.path.join(save_dir, 'tensors-%d.pkl' % split_id), 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

        print("preprocessing data tensor takes %.3f seconds" % (time.time() - tt))
