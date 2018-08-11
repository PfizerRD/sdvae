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

from cmd_args import cmd_args
cmd_args.grammar_file = ROOT + "/../dropbox/context_free_grammars/mol_zinc.grammar"

from mol_tree import AnnotatedTree2MolTree, get_smiles_from_tree
from mol_decoder import batch_make_att_masks
import cfg_parser as parser
from mol_tree import get_smiles_from_tree, Node
from tree_walker import OnehotBuilder, ConditionalDecoder
from attribute_tree_decoder import create_tree_decoder

from openeye import oechem
from openeye import oemolprop

def get_OEMol_from_smiles(smiles):
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, smiles)
    return mol
   
def canonicalize_smiles(smiles):
    mol = get_OEMol_from_smiles(smiles)
    smiles = oechem.OECreateCanSmiString(mol)
    return smiles

if __name__ == '__main__':
    one_hot_data = h5py.File(cmd_args.data_dump, 'r')
    x = one_hot_data.get('x')
    x = np.array(x)
    n_inputs = x.shape[0]

    raw_logits = np.transpose(x, [1, 0, 2])

    onehot_walker = OnehotBuilder()
    tree_decoder = create_tree_decoder()

    output_smiles_list = []
    property_list = []

    start_time = time.time()
    with open(cmd_args.prop_file, 'w') as prop_file:
        for i in range(raw_logits.shape[1]):
            if i % 500 == 0:
                print("Calculated properties for {:7d}/{:7d} compounds in {:.2f} min.".format(i, n_inputs, (time.time() - start_time)/60.0))
            pred_logits = raw_logits[:, i, :]
            walker = ConditionalDecoder(np.squeeze(pred_logits), use_random=False) # True doesn't help when you have a one-hot vector
            new_t = Node('smiles')
            try:
                tree_decoder.decode(new_t, walker)
                sampled = get_smiles_from_tree(new_t)
                #sampled = canonicalize_smiles(sampled)

                mol = oechem.OEGraphMol()
                oechem.OESmilesToMol(mol, sampled)
                sampled = oechem.OECreateCanSmiString(mol)
                result = oemolprop.OEGetXLogPResult(mol)
                if (result.IsValid()):
                    prop_file.write("{:.3f}\n".format(result.GetValue()))
                else:
                    prop_file.write("{:.3f}\n".format(0.0))
            except Exception as ex:
                if not type(ex).__name__ == 'DecodingLimitExceeded':
                    print('Warning, decoder failed with', ex)
            
    print("Done in {:.2f} min.".format((time.time()-start_time)/60.0))
    