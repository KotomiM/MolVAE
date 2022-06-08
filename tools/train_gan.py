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
from src.core.moledataloader import MoleDataLoader, MolTreeFolder, PairTreeFolder
from src.models import DiffVAE, ScaffoldGAN

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
#dataloader.preprocess()
moledataloader.load_preprocessed()

print('Train GAN: data loaded', params.train_set)
# vocab parameters passing to model
params.vocab = moledataloader.vocab
params.atom_vocab = moledataloader.atom_vocab
model = DiffVAE(params).cuda()
GAN = ScaffoldGAN(model, params.disc_hidden, beta=params.beta, gumbel=params.gumbel).cuda()

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

for param in GAN.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))
print("GAN #Params: %dK" % (sum([x.nelement() for x in GAN.parameters()]) / 1000,))

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

optimizer = optim.Adam(model.parameters(), lr=params.lr)
optimizerG = optim.Adam(model.parameters(), lr=params.gan_lrG, betas=(0, params.beta)) #generator is model parameter!
optimizerD = optim.Adam(GAN.netD.parameters(), lr=params.gan_lrD, betas=(0, params.beta))

scheduler = lr_scheduler.ExponentialLR(optimizer, params.anneal_rate)
scheduler.step()
"""
if params.load_epoch >= 0:
    print('continuing from checkpoint ' + params.load_model)
    model_state, optimizer_state, total_step, beta = torch.load(params.load_model)
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
else:
"""
total_step = 0
beta = params.beta

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))
assert params.gan_batch_size <= params.batch_size
#num_epoch = (params.train_epoch - params.load_epoch - 1) * (params.diter + 1) * 10
num_epoch = (params.train_epoch) * (params.diter + 1) * 10

meters = np.zeros(7)
if not os.path.exists(params.save_dir):
    os.makedirs(params.save_dir)

x_loader = PairTreeFolder(params.train_set, params.vocab, params.gan_batch_size, num_workers=4, y_assm=False, replicate=num_epoch)
x_loader = iter(x_loader)
y_loader = MolTreeFolder(params.ymols, params.vocab, params.gan_batch_size, num_workers=4, assm=False, replicate=num_epoch)
y_loader = iter(y_loader)


for epoch in range(params.train_epoch):
    meters = np.zeros(7)
    main_loader = PairTreeFolder(params.train_set, params.vocab, params.batch_size, num_workers=4)

    for it, batch in enumerate(main_loader):
        #1. Train encoder & decoder
        model.zero_grad()
        x_batch, y_batch = batch
        try:
            loss, kl_div, wacc, tacc, sacc = model(x_batch, y_batch, params.kl_lambda)
            loss.backward()
        except Exception as e:
            print(e)
            continue
        nn.utils.clip_grad_norm_(model.parameters(), params.clip_norm)
        optimizer.step()

        #2. Train discriminator
        for i in range(params.diter):
            GAN.netD.zero_grad()
            x_batch, _ = next(x_loader)
            y_batch = next(y_loader)
            d_loss, gp_loss = GAN.train_D(x_batch, y_batch, model)
            optimizerD.step()

        #3. Train generator (ARAE fashion)
        model.zero_grad()
        GAN.zero_grad()
        x_batch, _ = next(x_loader)
        y_batch = next(y_loader)
        g_loss = GAN.train_G(x_batch, y_batch, model)
        nn.utils.clip_grad_norm_(model.parameters(), params.clip_norm)
        optimizerG.step()
        
        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100, d_loss, g_loss, gp_loss])

        if total_step % params.print_iter == 0:
            meters /= params.print_iter
            print("KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, Disc: %.4f, Gen: %.4f, GP: %.4f, PNorm: %.2f, %.2f, GNorm: %.2f, %.2f" % (meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], meters[6], param_norm(model), param_norm(GAN.netD), grad_norm(model), grad_norm(GAN.netD)))
            sys.stdout.flush()
            meters *= 0
        
        if params.save_iter >= 0 and total_step % params.save_iter == 0:
            n_iter = total_step // params.save_iter - 1
            torch.save(model.state_dict(), params.save_dir + "/" + params.model + "_" + params.rnn_type+ "_model.iter-" + str(n_iter))
            torch.save(GAN.state_dict(), params.save_dir + "/" + params.model + "_" + params.rnn_type+ "_GAN.iter-" + str(n_iter))
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

    scheduler.step()
    if params.save_iter == -1:
        torch.save(model.state_dict(), params.save_dir + "/" + params.model + "_" + params.rnn_type+ "_model.iter-" + str(epoch))
        torch.save(GAN.state_dict(), params.save_dir + "/" + params.model + "_" + params.rnn_type+ "_GAN.iter-" + str(epoch))
        scheduler.step()
        print("learning rate: %.6f" % scheduler.get_lr()[0])
    
