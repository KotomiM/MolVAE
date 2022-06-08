import os, sys
#print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
from src.core.parameters import Parameters
from src.core.props import *
import moses

def convert(x):
    return None if x == "None" else x

def cal_succ_div(data, params):
    all_div = []
    n_succ = 0
    for i in range(0, len(data), params.num_decode):
        set_x = set([x[0] for x in data[i:i+params.num_decode]])
        assert len(set_x) == 1

        good = [convert(y) for x,y,sim,prop in data[i:i+params.num_decode] if sim >= params.sim_delta and prop >= params.prop_delta]
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
    return all_div, n_succ

def cal_impro(data, params):
    all_logp = []

    for i in range(0, len(data), params.num_decode):
        set_x = set([x[0] for x in data[i:i+params.num_decode]])
        assert len(set_x) == 1

        good = [(sim,logp) for _,y,sim,logp in data[i:i+params.num_decode] if 1 > sim >= params.delta and '.' not in y]
        if len(good) > 0:
            sim,logp = max(good, key=lambda x:x[1])
            all_logp.append(max(0,logp))
        else:
            all_logp.append(0.0) #No improvement

    all_logp = np.array(all_logp)
    return all_logp

def eval_logp(params):
    assert params.res_file
    print(params.res_file)
    res_f = open(params.res_file, 'r')
    score_list = []
    for line in res_f.readlines():
        x,y = line.split()[:2]
        if y == "None": y = None
        sim = similarity(x, y)
        try:
            prop = penalized_logp(y) - penalized_logp(x)
            score_list.append((x, y, float(sim), float(prop)))
        except Exception as e:
            score_list.append((x, y, float(sim), float(0.0)))

    all_div, n_succ = cal_succ_div(score_list, params)
    all_logp = cal_impro(score_list, params)
    print("Logp task")
    print('Evaluated on %d samples' % (len(score_list) // params.num_decode,))
    print("Diversity  Avg: " + str(np.mean(all_div)) + "  Std: " + str(np.std(all_div)))
    print("Improvement  Avg: " + str(np.mean(all_logp)) + " Std: " + str(np.std(all_logp)))
    res_f.close()

def eval_qed(params):
    assert params.res_file
    res_f = open(params.res_file, 'r')
    score_list = []
    for line in res_f.readlines():
        x,y = line.split()
        if y == "None": y = None
        sim2D = similarity(x, y)
        try:
            score_list.append((x, y, float(sim2D), float(qed(y))))
        except Exception as e:
            score_list.append((x, y, float(sim2D), float(0.0)))
    n_mols = len(score_list) / params.num_decode
    all_div, n_succ = cal_succ_div(score_list, params)
    print("QED task")
    print('Evaluated on %d samples' % (len(score_list) // params.num_decode,))
    print("Diversity  Avg: " + str(np.mean(all_div)) + "  Std: " + str(np.std(all_div)))
    print("Successful Rate: " + str(n_succ / n_mols))    

def eval_drd2(args):
    assert args.res_file
    res_f = open(args.res_file, 'r')
    score_list = []
    for line in res_f.readlines():
        x,y = line.split()
        if y == "None": y = None
        sim2D = similarity(x, y)
        try:
            score_list.append((x, y, float(sim2D), float(drd2(y))))
        except Exception as e:
            score_list.append((x, y, float(sim2D), float(0.0)))
    n_mols = len(score_list) / args.num_decode
    all_div, n_succ = cal_succ_div(score_list, args)
    print("DRD2 task")
    print("Diversity  Avg: " + str(np.mean(all_div)) + "  Std: " + str(np.std(all_div)))
    print("Successful Rate: " + str(n_succ / n_mols))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', required=True)
    parser.add_argument('--result', type=str, default='', required=True)
    args = parser.parse_args()
    print('load config from: ', args.config)
    params = Parameters()
    params.load(args.config)
    params.res_file = args.result
    print('params: ', params)

    if params.test_task == "generation":
        assert params.res_file
        res_f = open(params.res_file, 'r')
        res_list = [f.strip() for f in res_f.readlines()]
        if params.test_file:
            test_f = open(params.test_file, 'r')
            test_list = [f.strip() for f in test_f.readlines()]
            metrics = moses.get_all_metrics(res_list, test=test_list)
        else:
            metrics = moses.get_all_metrics(res_list)
        print("Generation Task")
        print(metrics)
  
    if params.test_task == "translation":
        print("Translation Task")
        if params.dataset == "qed":
            eval_qed(params)
        elif params.dataset == "drd2":
            eval_drd2(params)
        elif params.dataset == "logp":
            eval_logp(params)

    
