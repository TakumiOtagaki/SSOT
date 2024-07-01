import re
import sys
import numpy as np

param_file = "/large/otgk/SSOT/scripts/modules/mcCaskill/ViennaRNA/misc/rna_turner2004.par"


class RNAParams:
    def __init__(self):
        self.stack = dict()
        self.bulge = dict()
        self.hairpin = dict()
        self.internal = dict()
        self.ml_params = dict()
        self.stack_bases = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA', 'NN']
        self.limit_length = 30
        self.KT = 0.61632 # kcal/mol

def parse_rna_params(file_path):
    params = RNAParams()
    current_section = None
    stack_bases = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA', 'NN']

    section_patterns = {
        'stack': re.compile(r'#\s*stack$'),
        'bulge': re.compile(r'#\s*bulge$'),
        'hairpin': re.compile(r'#\s*hairpin$'),
        'internal': re.compile(r'#\s*interior$'),
        'ml_params': re.compile(r'#\s*ML_params$')
    }
    others_pattern = re.compile(r'#\s*')
    stack_bases = ['CG', 'GC', 'GU', 'UG', 'AU', 'UA', 'NN']
    internal_length = 0 
    hairpin_length = 0
    bulge_length = 0
    stack_counter = 0


    with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            line = line.strip()
            # if /* ... */ comment, ignore: start with '/*' and end with '*/'
            if line.startswith('/*'):
                while not line.endswith('*/'):
                    line = next(file).strip()
                continue
            matched = False
            if not line or line.startswith('#'):
                # 末尾の /* ... */ コメントを除去
                line = line.split('/*')[0].strip()
                for section, pattern in section_patterns.items():
                    if pattern.search(line):
                        current_section = section
                        matched = True
                        break
                if not matched:
                    current_section = 'others'
                continue
            print("current_section: ", current_section) 
            
            if current_section == "others":
                continue
            if current_section:
                # コメントを除去
                values = list(map(float, line.split("/*")[0].split()))
                # print("values: ", values)
                if current_section == 'stack':
                    left_base = stack_bases[stack_counter]
                    for i, value in enumerate(values):
                        right_base = stack_bases[i]
                        if left_base not in params.stack:
                            params.stack[left_base] = dict()
                        params.stack[left_base][right_base] = value
                    stack_counter += 1


                if current_section == 'bulge':
                    for value in values:
                        params.bulge[bulge_length] = value
                        bulge_length += 1

                elif current_section == 'hairpin':
                    for value in values:
                        params.hairpin[hairpin_length] = value
                        hairpin_length += 1
                elif current_section == 'internal':
                    for value in values:
                        params.internal[internal_length] = value
                        internal_length += 1
                elif current_section == 'ml_params':
                    params.ml_params["a"] = values[2] + values[5]
                    params.ml_params["b"] = values[3]
                    params.ml_params["c"] = values[4]
                    
                

    return params

def inf2large(prm, large=1e6):
    for key in prm.bulge:
        if prm.bulge[key] == np.float64("inf"):
            prm.bulge[key] = large
    for key in prm.hairpin:
        if prm.hairpin[key] == np.float64("inf"):
            prm.hairpin[key] = large
    for key in prm.internal:
        if prm.internal[key] == np.float64("inf"):
            prm.internal[key] = large
    for key in prm.stack:
        for key2 in prm.stack[key]:
            if prm.stack[key][key2] == np.float64("inf"):
                prm.stack[key][key2] = large
    return prm

def scale_params(prm):
    # 1 の order にする
    scaler = 1e-2
    for key in prm.bulge:
        prm.bulge[key] = prm.bulge[key] * scaler
    for key in prm.hairpin:
        prm.hairpin[key] = prm.hairpin[key] * scaler
    for key in prm.internal:
        prm.internal[key] = prm.internal[key] * scaler
    for key in prm.stack:
        for key2 in prm.stack[key]:
            prm.stack[key][key2] = prm.stack[key][key2] * scaler
    prm.ml_params["a"] = prm.ml_params["a"] * scaler
    prm.ml_params["b"] = prm.ml_params["b"] * scaler
    prm.ml_params["c"] = prm.ml_params["c"] * scaler
    return prm

if __name__ == "__main__":
    # 使用例
    file_path = param_file
    params = parse_rna_params(file_path)

    # Display parsed data for verification
    print("Stacking Energies:")
    for bp1, energies in params.stack.items():
        for bp2, energy in energies.items():
            print(f"{bp1}-{bp2}: {energy}")

    print("\nBulge Energies:")
    for length, energy in params.bulge.items():
        print(f"Length {length}: {energy}")

    print("\nHairpin Energies:")
    for length, energy in params.hairpin.items():
        print(f"Length {length}: {energy}")

    print("\nInternal Loop Energies:")
    for length, energy in params.internal.items():
        print(f"Length {length}: {energy}")

    print("\nMultiloop Parameters:")
    print(params.ml_params)