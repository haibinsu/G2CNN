
#! /usr/bin/env python
#
# Copyright (C) 2015-2016 Rich Lewis <rl403@cam.ac.uk>
# License: 3-clause BSD

"""
## skchem.descriptors.atom
Module specifying atom based descriptor generators.
"""

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors, rdPartialCharges
from rdkit.Chem.rdchem import HybridizationType

import os
import pandas as pd
from .correlations import *




def resource(*args):

    """ passes a file path for a data resource specified """

    return os.path.join(os.path.dirname(__file__), *args)

def get_periodic_table(scheme):
    if scheme == 3:
        PERIODIC_TABLE = pd.read_excel(resource('atom_data_scaled.xlsx'), index_col=0)
    else:
        PERIODIC_TABLE = pd.read_excel(resource('atom_data.xlsx'), index_col=0)
    return PERIODIC_TABLE



# Get the periodic table of current correlation scheme
PERIODIC_TABLE = get_periodic_table(pass_corr_scheme())



def element(a):

    """ Return the element """

    return a.GetSymbol()


def is_element(a, symbol='C'):

    """ Is the atom of a given element """
    return element(a) == symbol

#element_features = {'is_{}'.format(e): functools.partial(is_element, symbol=e)
#                    for e in ORGANIC}


def is_h_acceptor(a):

    """ Is an H acceptor? """

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return idx in [i[0] for i in Lipinski._HAcceptors(m)]


def is_h_donor(a):

    """ Is an H donor? """

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return idx in [i[0] for i in Lipinski._HDonors(m)]


def is_hetero(a):

    """ Is a heteroatom? """

    m = a.GetOwningMol()
    idx = a.GetIdx()
    return idx in [i[0] for i in Lipinski._Heteroatoms(m)]


def atomic_number(a):

    """ Atomic number of atom """

    return a.GetAtomicNum()


def atomic_mass(a):

    """ Atomic mass of atom """

    return a.GetMass()


def explicit_valence(a):

    """ Explicit valence of atom """
    return a.GetExplicitValence()


def implicit_valence(a):

    """ Implicit valence of atom """

    return a.GetImplicitValence()


def valence(a):

    """ returns the valence of the atom """

    return explicit_valence(a) + implicit_valence(a)


def formal_charge(a):

    """ Formal charge of atom """

    return a.GetFormalCharge()


def is_aromatic(a):

    """ Boolean if atom is aromatic"""

    return a.GetIsAromatic()


def num_implicit_hydrogens(a):

    """ Number of implicit hydrogens """

    return a.GetNumImplicitHs()


def num_explicit_hydrogens(a):

    """ Number of explicit hydrodgens """

    return a.GetNumExplicitHs()


def num_hydrogens(a):

    """ Number of hydrogens """

    return num_implicit_hydrogens(a) + num_explicit_hydrogens(a)


def is_in_ring(a):

    """ Whether the atom is in a ring """

    return a.IsInRing()


def crippen_log_p_contrib(a):

    """ Hacky way of getting logP contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return Crippen._GetAtomContribs(m)[idx][0]


def crippen_molar_refractivity_contrib(a):

    """ Hacky way of getting molar refractivity contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return Crippen._GetAtomContribs(m)[idx][1]


def tpsa_contrib(a):

    """ Hacky way of getting total polar surface area contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors._CalcTPSAContribs(m)[idx]


def labute_asa_contrib(a):

    """ Hacky way of getting accessible surface area contribution. """

    idx = a.GetIdx()
    m = a.GetOwningMol()
    return rdMolDescriptors._CalcLabuteASAContribs(m)[0][idx]


def gasteiger_charge(a, force_calc=False):

    """ Hacky way of getting gasteiger charge """
    """
    res = a.GetDoubleProp('_GasteigerCharge', None)
    if res and not force_calc:
        return float(res)
    else:
        m = a.GetOwningMol()
        rdPartialCharges.ComputeGasteigerCharges(m)
        return float(a.props['_GasteigerCharge'])
    """
    idx = a.GetIdx()
    m = a.GetOwningMol()
    # rdPartialCharges.ComputeGasteigerCharges(m)
    AllChem.ComputeGasteigerCharges(m)
    return m.GetAtomWithIdx(idx).GetDoubleProp('_GasteigerCharge')


def atomic_volume(a):

    """ Van der Waals volume of atom """

    return PERIODIC_TABLE.loc[a.GetAtomicNum(),'van_der_waals_volume']


def atomic_polarisability(a):

    """ Atomic polarisability of atom """

    return PERIODIC_TABLE.loc[a.GetAtomicNum(),'atomic_polarisability']

def electron_affinity(a):

    """ Electron affinity of atom """

    return PERIODIC_TABLE.loc[a.GetAtomicNum(),'electron_affinity']

def pauling_electronegativity(a):

    # return a.pauling_electronegativity
    return PERIODIC_TABLE.loc[a.GetAtomicNum(),'pauling_electronegativity']


def first_ionization(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), 'first_ionisation_energy']


def group(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), 'group']


def period(a):

    return PERIODIC_TABLE.loc[a.GetAtomicNum(), 'period']


def is_hybridized(a, hybrid_type=HybridizationType.SP3):

    """ Hybridized as type hybrid_type, default SP3 """

    return str(a.GetHybridization()) == str(hybrid_type)

#hybridization_features = {'is_' + n + '_hybridized': functools.partial(
#    is_hybridized, hybrid_type=n)
#                          for n in HybridizationType.names}

ATOM_FEATURES = {
    'atomic_number': atomic_number,
    'atomic_mass': atomic_mass,
    'formal_charge': formal_charge,
    'gasteiger_charge': gasteiger_charge,
    'pauling_electronegativity': pauling_electronegativity,
    'first_ionisation': first_ionization,
    'group': group,
    'period': period,
    'valence': valence,
    'is_aromatic': is_aromatic,
    'num_hydrogens': num_hydrogens,
    'is_in_ring': is_in_ring,
    'log_p_contrib': crippen_log_p_contrib,
    'molar_refractivity_contrib': crippen_molar_refractivity_contrib,
    'is_h_acceptor': is_h_acceptor,
    'is_h_donor': is_h_donor,
    'is_heteroatom': is_hetero,
    'total_polar_surface_area_contrib': tpsa_contrib,
    'total_labute_accessible_surface_area': labute_asa_contrib,
}
# ATOM_FEATURES.update(element_features)
# ATOM_FEATURES.update(hybridization_features)