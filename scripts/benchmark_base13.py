import os
import sys
import glob
from subprocess import check_output, PIPE
from tqdm import tqdm

# usage: python benchmark.py dataset_name
dataset = sys.argv[1] # wiki_ts_200M_uint64
db_dataset=f"./data/{dataset}"
assert os.path.exists(db_dataset), db_dataset
# query="../data/{dataset}_queries_10M_in"
for query in tqdm(sorted(glob.glob(db_dataset+"_*_test"))):
    query_name = os.path.basename(query)
    query_train_name = query_name[:-len('_test')]+'_train'
    cmd=f"./build/benchmark {db_dataset} {query} 5 rmi ./rmi_data/base13/{query_train_name}_3_PARAMETERS"
    print("====>running ", cmd, file=sys.stderr)
    outs = check_output(cmd, shell=True, universal_newlines=True)
    outs = outs.strip().split('\n')[-1]
    assert outs.startswith("RESULT"), outs
    print(outs[len("RESULT: "):]+","+query_name)
