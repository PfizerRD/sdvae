import os
import sys
import numpy as np
import math
import random
import h5py
import time

import logging
logger = logging.getLogger(__name__)

from tqdm import tqdm

ROOT = '/home/sreshv/Research/Projects/AI_Library_Design/sdvae_original/mol_vae'
os.chdir(ROOT + "/data_processing/")

sys.path.append(ROOT + '/mol_common')
sys.path.append(ROOT + '/cfg_parser')
sys.path.append(ROOT + '/mol_decoder')
sys.path.append(ROOT + '/pytorch_eval')

from cmd_args import cmd_args
cmd_args.grammar_file = ROOT + "/../dropbox/context_free_grammars/mol_zinc.grammar"

import argparse


from mol_tree import AnnotatedTree2MolTree, get_smiles_from_tree
from mol_decoder import batch_make_att_masks
from cfg_parser import renumber_independent_rings
import cfg_parser as parser
from mol_tree import get_smiles_from_tree, Node
from tree_walker import OnehotBuilder, ConditionalDecoder
from attribute_tree_decoder import create_tree_decoder
from att_model_proxy import AttMolProxy

from openeye import oechem, oedepict
from IPython.display import display_png, HTML

def get_OEMol_from_smiles(smiles):
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    return mol
   
def canonicalize_smiles(smiles):
    mol = get_OEMol_from_smiles(smiles)
    smiles = oechem.OECreateCanSmiString(mol)
    return smiles

def depict_smiles(smiles):
    mol = get_OEMol_from_smiles(smiles)
    oedepict.OEPrepareDepiction(mol)
    opts = oedepict.OE2DMolDisplayOptions()
    disp = oedepict.OE2DMolDisplay(mol, opts)
    img_string = oedepict.OERenderMoleculeToString("png", disp)
    return img_string

def draw_smiles_grid(smiles_list, width=900, height=200, rows=1, cols=10):
    image = oedepict.OEImage(width, height, oechem.OETransparentColor)
    grid = oedepict.OEImageGrid(image, rows, cols)
    opts = oedepict.OE2DMolDisplayOptions(grid.GetCellWidth(), grid.GetCellHeight(), oedepict.OEScale_AutoScale)
    
    assert len(smiles_list) == rows*cols
    
    for smi, cell in zip(smiles_list, grid.GetCells()):
        mol = get_OEMol_from_smiles(smi)
        oedepict.OEPrepareDepiction(mol)
        disp = oedepict.OE2DMolDisplay(mol, opts)
        oedepict.OERenderMolecule(cell, disp)
    
    img_string = oedepict.OEWriteImageToString("png", image)
    return img_string

def save_img_str_to_file(img_string, file):
    with open(file, "wb") as fh:
        fh.write(img_string)
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Test Reconstruction Accuracy")
    parser.add_argument("model_name", type=str, help="name of model")
    parser.add_argument("-n", type=int, help="size of test_list", default=-1)
    parser.add_argument("-ae_type", default='vae')
    args = parser.parse_args()
    cmd_args.ae_type = args.ae_type
    
    if args.model_name == 'zinc_kl_sum':
        cmd_args.saved_model = ROOT + "/../dropbox/results/zinc/{:s}.model".format(args.model_name)
    else:
        cmd_args.saved_model = ROOT + "/../dropbox/results/sreshv/{:s}/epoch-best.model".format(args.model_name)
    
    glp1_file = ROOT + "/../dropbox/data/glp1/glp1_smiles_good.smi" 
    test_list = []
    with open(glp1_file, 'r') as input_file:
        for smiles in input_file:
            test_list.append(canonicalize_smiles(smiles.strip()))

    test_list = list(set(test_list))
    print("{:d} SMILES read!".format(len(test_list)))

    start_time = time.time()
    proxy = AttMolProxy()
    if args.n > 0:
        test_molecules = test_list[:args.n]
    else:
        test_molecules = test_list
        
    decoded_list = []
    for smi_no, smi in enumerate(test_molecules):
        z_mean = proxy.encode([smi])
        decoded_smi = proxy.decode(z_mean, use_random=True)
        sys.stdout.write("{:4d}/{:4d} successfully decoded!\r".format(smi_no+1, len(test_molecules)))
        sys.stdout.flush()
        decoded_list.append(decoded_smi[0])
    print("{:4d} successfully decoded!".format(len(test_molecules)))
    
    #z_mean = proxy.encode(test_molecules)
    #print(z_mean.shape)
    #decoded_list = proxy.decode(z_mean, use_random=True)
    print("{:d} SMILES encoded and decoded in {:.2f} sec.".format(len(test_molecules), time.time() - start_time))

    original_list = [canonicalize_smiles(x) for x in test_molecules]
    decoded_list = [canonicalize_smiles(x) for x in decoded_list]
    incorrect_molecules = []
    for a, b in zip(original_list, decoded_list):
        if a != b:
            incorrect_molecules.append(a)

    print("{:d} of {:d} ({:.2f}%) correctly reproduced.".format(
        len(original_list) - len(incorrect_molecules),
        len(original_list),
        (len(original_list) - len(incorrect_molecules))*100.0/(len(original_list))
    ))

    img_string = draw_smiles_grid(original_list[:9], width=4*300, height=4*300, rows=3, cols=3)
    OUTPUT_DIR = '/home/sreshv/scratch/sdvae_results'
    save_img_str_to_file(img_string, OUTPUT_DIR + "/{}_original.png".format(args.model_name))

    img_string = draw_smiles_grid(decoded_list[:9], width=4*300, height=4*300, rows=3, cols=3)
    OUTPUT_DIR = '/home/sreshv/scratch/sdvae_results'
    save_img_str_to_file(img_string, OUTPUT_DIR + "/{}_decoded.png".format(args.model_name))