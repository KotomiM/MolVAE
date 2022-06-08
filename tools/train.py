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
import tqdm

import rdkit
from src.core.parameters import Parameters
from src.core.moledataloader import MoleDataLoader
from src.models.model_factory import ModelFactory

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', required=True)
args = parser.parse_args()

print('load config from: ', args.config)

params = Parameters()
params.load(args.config)
print(params)
moledataloader = MoleDataLoader(params)
moledataloader.load_vocab_set()
#dataloader.preprocess()
#moledataloader.load_preprocessed()

print('Train: data loaded')
# vocab parameters passing to model
params.vocab = moledataloader.vocab
params.atom_vocab = moledataloader.atom_vocab

print(params.vocab.size())

model = ModelFactory.get_model(params)
model = model.cuda()

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=params.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, params.anneal_rate)

if params.load_model != "" and params.load_epoch >= 0:
    print('continuing from checkpoint ' + params.load_model)
    model_state, optimizer_state, total_step, beta = torch.load(params.load_model)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
else:
    print('train from epoch 0')
    total_step = 0
    beta = params.beta
    params.load_epoch = -1

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

if not os.path.exists(params.model_save_dir):
    os.makedirs(params.model_save_dir)

# model meters difference
if params.model == 'hiervae' or params.model == 'hiervgnn':
    meters = np.zeros(6)
    for epoch in range(params.load_epoch + 1, params.train_epoch):
        dataset = moledataloader.get_dataset()

        for batch in dataset:
            total_step += 1
            model.zero_grad()
            loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params.clip_norm)
            optimizer.step()

            meters = meters + np.array([kl_div, loss.item(), wacc.cpu() * 100, iacc.cpu() * 100, tacc.cpu() * 100, sacc.cpu() * 100])

            if total_step % params.print_iter == 0:
                meters /= params.print_iter
                print("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))
                sys.stdout.flush()
                meters *= 0
            
            if params.save_iter >= 0 and total_step % params.save_iter == 0:
                n_iter = total_step // params.save_iter - 1
                torch.save(model.state_dict(), params.model_save_dir + "/" + params.model + "_" + params.rnn_type+ "." + str(n_iter))
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

        del dataset
        if params.save_iter == -1:
            torch.save(model.state_dict(), params.model_save_dir + "/" + params.model + "_" + params.rnn_type+ ".epoch" + str(epoch))
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])



if params.model == 'vjtnn':
    meters = np.zeros(4)
    for epoch in range(params.load_epoch + 1, params.train_epoch):
        dataset = moledataloader.get_dataset()

        for batch in dataset:
            total_step += 1
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc = model(*batch, beta=beta)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params.clip_norm)
            optimizer.step()

            meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])
            if total_step % params.print_iter == 0:
                meters /= params.print_iter
                print( "[%d] KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
                sys.stdout.flush()
                meters *= 0
            
            if params.save_iter >= 0 and total_step % params.save_iter == 0:
                n_iter = total_step // params.save_iter - 1
                torch.save(model.state_dict(), params.model_save_dir + "/" + params.model + "_" + params.rnn_type+ "." + str(n_iter))
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

        del dataset
        if params.save_iter == -1:
            torch.save(model.state_dict(), params.model_save_dir + "/" + params.model + "_" + params.rnn_type+ ".epoch" + str(epoch))
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])


if params.model == 'jtvae':
    meters = np.zeros(4)
    for epoch in range(params.load_epoch + 1, params.train_epoch):
        dataset = moledataloader.get_dataset()

        for batch in dataset:
            total_step += 1
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc = model(batch, beta=beta)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params.clip_norm)
            optimizer.step()

            meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])
            if total_step % params.print_iter == 0:
                meters /= params.print_iter
                print( "[%d] KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, meters[0], meters[1], meters[2], meters[3], param_norm(model), grad_norm(model)))
                sys.stdout.flush()
                meters *= 0
            
            if params.save_iter >= 0 and total_step % params.save_iter == 0:
                n_iter = total_step // params.save_iter - 1
                torch.save(model.state_dict(), params.model_save_dir + "/" + params.model + "_" + params.rnn_type+ "." + str(n_iter))
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

        del dataset
        if params.save_iter == -1:
            torch.save(model.state_dict(), params.model_save_dir + "/" + params.model + "_" + params.rnn_type+ ".epoch" + str(epoch))
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

