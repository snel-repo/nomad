from collections import namedtuple

"""
Classes that organize groups of tensors into namedtuples. 
Names and uses should be self-explanatory.
"""

AlignInput = namedtuple(
    'AlignInput', [
        'day0_train_data', 
        'day0_valid_data', 
        'dayk_train_data', 
        'dayk_valid_data', 
        'day0_train_inds', 
        'day0_valid_inds', 
        'dayk_train_inds', 
        'dayk_valid_inds', 
    ])

SingleModelOutput = namedtuple(
    'SingleModelOutput', [
        'rates',
        'factors',
        'gen_inputs',
        'gen_states',
    ])

AlignmentOutput = namedtuple(
    'AlignmentOutput', [
        'day0',
        'dayk',
        'aligned',
    ])