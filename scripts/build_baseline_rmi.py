import argparse
from subprocess import Popen, PIPE, check_output
import os
import json
import re

def parse_l0_params(file):
    with open(file, 'r') as fin:
        cnt = 0
        res = []
        for line in fin:
            match = re.match(r"const double L0_PARAMETER(\d+) = ([+-]?([0-9]*[.])?[0-9]+);", line)
            if match:
                assert cnt == int(match.group(1)), (cnt, match.group())
                print(match.group(), match.group(2))
                res.append(float(match.group(2)))
                cnt += 1
    return res

parser = argparse.ArgumentParser()
parser.add_argument('--param-grid', required=True)
parser.add_argument('--data', required=True)
parser.add_argument('--output-path', required=True)
parser.add_argument('--RMI-path', default='../SOSD/RMI')
args = parser.parse_args()
if not os.path.exists('tmp'): os.makedirs('tmp')
if not os.path.exists(args.output_path): os.makedirs(args.output_path)
pg = json.load(open(args.param_grid))

rmi_bin = args.RMI_path+'/target/release/rmi'
param_grid = args.param_grid
data = args.data
check_output(f"cd {args.RMI_path} && cargo build --release", shell=True)
check_output(f'{rmi_bin} {data} --param-grid {param_grid} -d ./tmp --threads 4 --zero-build-time', shell=True)
for i in pg['configs']:
    name = i['namespace']
    arch = i['layers']
    l0_params = parse_l0_params(name+'_data.h')
    cmd = f"python scripts/convert_data.py tmp/{name}_L1_PARAMETERS {args.output_path}/{name}_PARAMETERS {arch} {' '.join(map(str, l0_params))}"
    print(cmd)
    check_output(cmd, shell=True)
    check_output(f"mv {name}* ./tmp/", shell=True)
