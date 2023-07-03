from argparse import Namespace
from typing import List, Tuple, Union

from rdkit import Chem
import torch
import numpy as np
#from descriptastorus.descriptors import rdNormalizedDescriptors
import pandas as pd
import os
"""
## skchem.descriptors.atom
Module specifying atom based descriptor generators.
"""

scale_scheme=3

def pass_corr_scheme():
    return scale_scheme

from .atomfeatures import *

CORR_FEATURES = {
    #'atomic_mass': atomic_mass, # Will cause ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    'atomic_volume': atomic_volume,
    'atomic_polarisability': atomic_polarisability,
    'electron_affinity': electron_affinity,
    'pauling_electronegativity': pauling_electronegativity,
    'first_ionisation': first_ionization,
    # 'formal_charge': formal_charge,
    # 'gasteiger_charge': gasteiger_charge,
}

def get_corr_scale_scheme(scheme=3):
    print(f"Current correlation scale scheme is scheme {scheme}")
    if scheme == 0:
        CORR_SCALES = {
           'atomic_volume': 0.0001, 'atomic_polarisability': 0.1,
           'electron_affinity': 1, 'pauling_electronegativity': 0.1,
           'first_ionisation': 0.01}
    elif scheme == 1:
        CORR_SCALES = {
           'atomic_volume': 0.001, 'atomic_polarisability': 0.01,
           'electron_affinity': 0.1, 'pauling_electronegativity': 0.1,
           'first_ionisation': 0.001}
    elif scheme == 2:
        CORR_SCALES = {
           'atomic_volume': 0.001, 'atomic_polarisability': 0.01,
           'electron_affinity': 0.1, 'pauling_electronegativity': 0.1,
           'first_ionisation': 0.01}
    elif scheme == 3:
        CORR_SCALES = {'atomic_volume': 1, 'atomic_polarisability': 1,
           'electron_affinity': 1, 'pauling_electronegativity': 1,
           'first_ionisation': 1}

    return CORR_SCALES, PERIODIC_TABLE


CORR_SCALES, PERIODIC_TABLE = get_corr_scale_scheme(scheme=scale_scheme)

def pass_corr_scheme():
    return CORR_FEATURES, CORR_SCALES


def atomic_correlations(smiles, CORR_FEATURES, CORR_SCALES, max_length=66):
    cmol = Chem.MolFromSmiles(smiles.split('.')[0])
    amol = Chem.MolFromSmiles(smiles.split('.')[1])

    corr_matrices = []
    for name, corr in CORR_FEATURES.items():
        corr_matrix = np.zeros((max_length,  max_length))
        corr_scale=CORR_SCALES[name]

        for i, catom in enumerate(cmol.GetAtoms()):
            for j, aatom in enumerate(amol.GetAtoms()):
                correlation = corr(catom) * corr(aatom)
                corr_matrix[j, i] = correlation * corr_scale

        corr_matrices.append(corr_matrix.tolist())

    c_corr_matrices = np.vstack(corr_matrices)
    a_corr_matrices = np.hstack(corr_matrices)

    # get atomic vectors for the whole molecule
    mol_corr = []
    for i, atom in enumerate(cmol.GetAtoms()):
        mol_corr.append(c_corr_matrices[:, i].tolist())
    for i, atom in enumerate(amol.GetAtoms()):
        mol_corr.append(a_corr_matrices[i].tolist())

    return mol_corr
