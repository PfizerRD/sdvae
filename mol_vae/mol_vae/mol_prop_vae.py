#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import numpy as np
import math
import random

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.append('%s/../mol_common' % os.path.dirname(os.path.realpath(__file__)))
from mol_util import rule_ranges, terminal_idxes, DECISION_DIM
from mol_tree import Node, get_smiles_from_tree, AnnotatedTree2MolTree, AnnotatedTree2Onehot
from cmd_args import cmd_args

sys.path.append('%s/../mol_decoder' % os.path.dirname(os.path.realpath(__file__)))
from mol_decoder import batch_make_att_masks, PerpCalculator, StateDecoder, PropCalculator

sys.path.append('%s/../mol_encoder' % os.path.dirname(os.path.realpath(__file__)))
from mol_encoder import CNNEncoder

sys.path.append('%s/../cfg_parser' % os.path.dirname(os.path.realpath(__file__)))
import cfg_parser as parser

def parse_smiles_with_cfg(smiles_file, grammar_file):
    grammar = parser.Grammar(cmd_args.grammar_file)

    smiles_list = []
    cfg_tree_list = []
    annotated_trees = []
    with open(smiles_file, 'r') as f:
        for row in tqdm(f):
            smiles = row.strip()
            smiles_list.append(smiles)
            ts = parser.parse(smiles, grammar)
            assert isinstance(ts, list) and len(ts) == 1
            annotated_trees.append(ts[0])
            n = AnnotatedTree2MolTree(ts[0])
            cfg_tree_list.append(n)
            st = get_smiles_from_tree(n)

            assert st == smiles

    return (smiles_list, cfg_tree_list, annotated_trees)


def get_encoder():
    if cmd_args.encoder_type == 'cnn':
        return CNNEncoder(max_len=cmd_args.max_decode_steps, latent_dim=cmd_args.latent_dim)
    else:
        raise ValueError('unknown encoder type %s' % cmd_args.encoder_type)


class MolPropVAE(nn.Module):
    def __init__(self):
        super(MolPropVAE, self).__init__()
        print('using propvae')
        self.latent_dim = cmd_args.latent_dim
        self.encoder = get_encoder()
        self.state_decoder = StateDecoder(max_len=cmd_args.max_decode_steps, latent_dim=cmd_args.latent_dim)
        self.perp_calc = PerpCalculator()
        self.prop_calc = PropCalculator(latent_dim=cmd_args.latent_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            eps = mu.data.new(mu.size()).normal_(0, cmd_args.eps_std)            
            if cmd_args.mode == 'gpu':
                eps = eps.cuda()
            eps = Variable(eps)
            
            return mu + eps * torch.exp(logvar * 0.5)            
        else:
            return mu

    def forward(self, x_inputs, true_binary, rule_masks, prop_inputs):        
        z_mean, z_log_var = self.encoder(x_inputs)

        z = self.reparameterize(z_mean, z_log_var)

        raw_logits = self.state_decoder(z)
        perplexity = self.perp_calc(true_binary, rule_masks, raw_logits)
        
        prop_loss = self.prop_calc(z_mean, prop_inputs) 

        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
        
        return (perplexity, cmd_args.kl_coeff * torch.mean(kl_loss), cmd_args.prop_coeff*prop_loss)

