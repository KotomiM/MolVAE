import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import rdkit.Chem.QED as QED
from rdkit import rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.six.moves import cPickle
from rdkit.six import iteritems

import numpy as np


def get_score(smile):
    if clf_model is None:
        load_model()

    mol = Chem.MolFromSmiles(smile)
    if mol:
        fp = fingerprints_from_mol(mol)
        score = clf_model.predict_proba(fp)[:, 1]
        return float(score)
    return 0.0

def fingerprints_from_mol(mol):
    fp = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    size = 2048
    nfp = np.zeros((1, size), np.int32)
    for idx,v in fp.GetNonzeroElements().items():
        nidx = idx%size
        nfp[0, nidx] += int(v)
    return nfp



def penalized_logp(s):
    if s is None: return -100.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle

def similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2) 

def qed(s):
    if s is None: return 0.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return 0.0
    return QED.qed(mol)

def drd2(s):
    if s is None: return 0.0
    if Chem.MolFromSmiles(s) is None:
        return 0.0
    return get_score(s)

def convert(x):
    return None if x == "None" else x

def cal_succ_div(score_list, args):
    all_div = []
    n_succ = 0
    for i in range(0, len(score_list), args.num_decode):
        set_x = set([x[0] for x in score_list[i:i+args.num_decode]])
        assert len(set_x) == 1

        good = [convert(y) for x,y,sim,prop in score_list[i:i+args.num_decode] if sim >= args.sim_delta and prop >= args.prop_delta]
        if len(good) == 0:
            continue

        good = list(set(good))
        if len(good) == 1:
            all_div.append(0.0)
            continue
        n_succ += 1
    
        div = 0.0
        tot = 0
        for i in range(len(good)):
            for j in range(i + 1, len(good)):
                div += 1 - similarity(good[i], good[j])
                tot += 1
        div /= tot
        all_div.append(div)

    all_div = np.array(all_div) 

def cal_impro(score_list, args):
    for i in range(0, len(score_list), args.num_decode):
        set_x = set([x[0] for x in score_list[i:i+args.num_decode]])
        assert len(set_x) == 1

        good = [(sim,logp) for _,_,sim,logp in score_list[i:i+args.num_decode] if 1 > sim >= args.delta]
        if len(good) > 0:
            sim,logp = max(good, key=lambda x:x[1])
            all_logp.append(max(0,logp))
        else:
            all_logp.append(0.0) #No improvement
    all_logp = np.array(all_logp)
    return all_logp