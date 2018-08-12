from __future__ import print_function
import sys
import os
import time
import argparse
import math
import itertools

from openeye.oechem import OEMol, OESmilesToMol, OECreateCanSmiString
from openeye.oemolprop import OEGetXLogP

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt, qed
from sascorer import calculateScore as calculateSAScore

from joblib import Parallel, delayed


def get_OEMol_from_smiles(smiles):
    mol = OEMol()
    OESmilesToMol(mol, smiles)
    return mol
   
def canonicalize_smiles(smiles):
    mol = get_OEMol_from_smiles(smiles)
    smiles = OECreateCanSmiString(mol)
    return smiles

def get_mw(smi):
    m = Chem.MolFromSmiles(smi)
    return MolWt(m)

def get_qed(smi):
    m = Chem.MolFromSmiles(smi)
    return qed(m)

def get_clogp(smi):
    m = get_OEMol_from_smiles(smi)
    return OEGetXLogP(m)

def get_sas(smi):
    m = Chem.MolFromSmiles(smi)
    sas_score = calculateSAScore(m)
    if type(sas_score) != float:
        print(sas_score)
        print(type(sas_score))
        raise ValueError
    return sas_score

def process_chunk(chunk):
    MW_list = []
    clogp_list = []
    QED_list = []
    SAS_list = []
    for smi in chunk:
        try:
            smi = canonicalize_smiles(smi)
            mw = get_mw(smi)
            clogp = get_clogp(smi)
            QED = get_qed(smi)
            sas = get_sas(smi)
        except:
            print("Could not calculate properties for {:s}".format(smi))
            continue
        MW_list.append(mw)
        clogp_list.append(clogp)
        QED_list.append(QED)
        SAS_list.append(sas)
    
    return (MW_list, clogp_list, QED_list, SAS_list)


def chunks(L, n):
    """Yield successive n-sized chunks from L."""
    for j in xrange(0, len(L), n):
        yield L[j:j + n]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Dataset Properties")
    parser.add_argument("smi_file")
    parser.add_argument("-n", default=2, type=int)
    args = parser.parse_args()
    
    prefix = os.path.splitext(args.smi_file)[0]
    
    smiles_list = []
    with open(args.smi_file, 'r') as input_file:
        for l in input_file:
            smiles_list.append(l.strip())
    print("Read {:d} smiles from {:s}".format(len(smiles_list), args.smi_file))
    print("Output prefix: {:s}".format(prefix))
        
    start_time = time.time()
    chunk_size=5000
    results = Parallel(n_jobs=args.n, verbose=50)(
        delayed(process_chunk)(smiles_list[start: start + chunk_size])
        for start in range(0, len(smiles_list), chunk_size)
    )
    zipped_results = zip(*results)
    MW_results = list(itertools.chain.from_iterable(zipped_results[0]))
    clogp_results = list(itertools.chain.from_iterable(zipped_results[1]))
    SAS_results = list(itertools.chain.from_iterable(zipped_results[2]))
    QED_results = list(itertools.chain.from_iterable(zipped_results[3]))
    #MW_results, clogp_results, QED_results, SAS_results = zip(*results)

    print("Finished calculation for {:d} molecules in {:.2f}s".format(len(MW_results), time.time() - start_time))
    
    #MW_results, clogp_results, QED_results, SAS_results = process_chunk(smiles_list)
    #print("Finished calculation for {:d} molecules in {:.2f}s".format(len(smiles_list), time.time() - start_time))
    
    print("Writing MW")
    with open(prefix+'_mw.csv', 'w') as output_file:
        for i in MW_results:
            output_file.write("{:.3f}\n".format(i))
    print("Writing ClogP")
    with open(prefix+'_clogp.csv', 'w') as output_file:
        for i in clogp_results:
            output_file.write("{:.3f}\n".format(i))
    print("Writing QED")
    with open(prefix+'_qed.csv', 'w') as output_file:
        for i in QED_results:
            output_file.write("{:.3f}\n".format(i))
    print("Writing SAS")
    with open(prefix+'_sas.csv', 'w') as output_file:
        for i in SAS_results:
            #print(i)
            output_file.write("{:.3f}\n".format(i))
