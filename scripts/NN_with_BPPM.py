"""
In this script,
 - we will optimize the Base Pair Probability Matrix (BPPM):
    - loss is the norm of target secondary structure matrix(0/1) - predicted secondary structure's BPPM
 - we encode the sequence into one-hot encoding
    - f_hairpin, f_stack, f_bulge, f_internal are modified to accept one-hot encoding.
    - we will use the McCaskill algorithm to calculate the BPPM
    - BPPM_ij > t (threshold) => i-j is a base pair
        - exp(- (BPPM_ij - t)^2) => i-j is a base pair  (??)
"""

import numpy as np
from modules.mcCaskill import mcCaskill, load_params
from modules.nussinov import onehot_differential_nussinov

# Load RNA parameters
param_file = "/large/otgk/SSOT/scripts/modules/mcCaskill/ViennaRNA/misc/rna_turner2004.par"
prm = load_params.parse_rna_params(param_file)

