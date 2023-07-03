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



# path = '/home/zhuangck/pycharm_projects/CMPNN/data/data_ILthermo/'
# cationDescriptors = pd.read_csv(path + 'Cation_features_2d_norm.csv')
# anionDescriptors = pd.read_csv(path + 'Anion_features_2d_norm.csv')

"""
# rdkit 2d features
def get_rdkit_2d(smiles):
    # generator = rdDescriptors.RDKit2D()
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    if len(smiles.split('.')) == 2:
        cation, anion = smiles.split('.')
    elif len(smiles.split('.')) == 3:
        cation = smiles.split('.')[0]
        bisdicyanamide = smiles.split('.')[1:]
        anion = '.'.join(bisdicyanamide)

    features=[]
    #cation_features = generator.process(cation)[1:]
    #anion_features = generator.process(anion)[1:]
    cation_df = cationDescriptors[cationDescriptors['Smiles'].isin([cation])]
    cation_features = cation_df[cation_df.columns[2:]].values.tolist()[0]

    anion_df = anionDescriptors[anionDescriptors['Smiles'].isin([anion])]
    anion_features = anion_df[anion_df.columns[2:]].values.tolist()[0]

    del_cfeatures = [16, 60, 93, 100, 117, 126, 128, 136, 137, 149, 151, 152, 156, 160, 163, 164, 165, 166, 168, 173, 174, 175, 179, 180, 192, 198]
    del_afeatures = [133, 137, 147, 162, 163, 165, 166, 175, 179, 180, 184, 187, 189, 198]

    new_cation_features=[ desc for i, desc in enumerate(cation_features) if i not in del_cfeatures]
    new_anion_features=[ desc for i, desc in enumerate(anion_features) if i not in del_afeatures]

    features.extend(new_cation_features)
    features.extend(new_anion_features)
    return features
"""


def atomic_correlations_self(i, atom, cation, anion, mol):
    # cation = Chem.MolFromSmiles(smiles.split('.')[0])
    # anion = Chem.MolFromSmiles(smiles.split('.')[1])
    all_corr = []

    cation_atoms = range(len(cation.GetAtoms()))
    anion_atoms = range(len(cation.GetAtoms()), len(mol.GetAtoms()))

    for name, corr in CORR_FEATURES.items():
        for ions in [cation_atoms, anion_atoms]:
            correlation = np.zeros(20, )
            for k, j in enumerate(ions):
                atom1 = mol.GetAtoms()[j]
                try:
                    correlation[k] = (corr(atom) * corr(atom1))
                    # scaled to about the same range as other features
                    if name in ['atomic_mass', 'atomic_volume', 'first_ionisation']:
                        correlation[k] = correlation[k] * 0.001
                    elif name in ['electron_affinity', 'pauling_electronegativity']:
                        correlation[k] = correlation[k] * 0.1
                    elif name in ['atomic_polarisability']:
                        correlation[k] = correlation[k] * 0.01
                    correlation[k] = (correlation[k] - CORR_SCALES[name][1]) / (
                                CORR_SCALES[name][0] - CORR_SCALES[name][1])
                except:
                    pass
            correlation_new = correlation.tolist()
            all_corr.extend(correlation_new)
    return all_corr



def atomic_correlations_coarse(i, atom, cation, anion, mol):
    all_corr = []
    cation_atoms = range(len(cation.GetAtoms()))
    anion_atoms = range(len(cation.GetAtoms()), len(mol.GetAtoms()))

    for name, corr in CORR_FEATURES.items():
        for ions in [cation_atoms, anion_atoms]:
            correlation = []
            correlation_coarse = []
            correlation_new = np.zeros(14, )
            for j in (ions):
                atom1 = mol.GetAtoms()[j]
                atom_corr = corr(atom) * corr(atom1)

                # scaled to about the same range as other features
                if name in ['atomic_mass', 'atomic_volume', 'first_ionisation']:
                    atom_corr = atom_corr * 0.001
                elif name in ['electron_affinity', 'pauling_electronegativity']:
                    atom_corr = atom_corr * 0.1
                elif name in ['atomic_polarisability']:
                    atom_corr = atom_corr * 0.01

                correlation.append((atom_corr - CORR_SCALES[name][1]) / (CORR_SCALES[name][0] - CORR_SCALES[name][1]))

            for k in range(0, len(correlation), 2):
                try:
                    atom_corr_new = (correlation[k] + correlation[k + 1]) / 2
                    correlation_coarse.append(atom_corr_new)
                except:
                    pass
            for l, corr_coarse in enumerate(correlation_coarse):
                try:
                    correlation_new[l] = corr_coarse
                except:
                    pass
            all_corr.extend(correlation_new)

    return all_corr