import os
import sys
import glob
from subprocess import Popen, PIPE
from tqdm import tqdm

# usage: python benchmark.py dataset_name
dataset = sys.argv[1] # wiki_ts_200M_uint64
db_dataset=f"./data/{dataset}"
assert os.path.exists(db_dataset), db_dataset
# query="../data/{dataset}_queries_10M_in"
for query in tqdm(sorted(glob.glob(db_dataset+"_*_test"))):
    query_name = os.path.basename(query)
    for i in range(3, 7):
        cmd=f"./build/benchmark {db_dataset} {query} 5 rmi ./rmi_data/baseline/{dataset}_{i}_PARAMETERS"
        print("====>running ", cmd, file=sys.stderr)
        outs, errs = Popen(cmd, shell=True, stdout=PIPE, universal_newlines=True).communicate()
        outs = outs.strip().split('\n')[-1]
        assert outs.startswith("RESULT"), outs
        print(outs[len("RESULT: "):]+","+query_name)
