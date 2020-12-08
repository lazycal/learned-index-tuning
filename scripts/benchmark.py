import os
import sys
import glob
from subprocess import Popen, PIPE
from tqdm import tqdm

db_dataset="./data/wiki_ts_200M_uint64"
assert os.path.exists(db_dataset), db_dataset
# query="../data/wiki_ts_200M_uint64_queries_10M_in"
for query in tqdm(sorted(glob.glob(db_dataset+"_*_test"))):
    query_name = os.path.basename(query)
    cmd=f"./build/benchmark {db_dataset} {query} ./rmi_data/baseline/wiki_ts_200M_uint64_0_L1_PARAMETERS"
    print("====>running ", cmd, file=sys.stderr)
    outs, errs = Popen(cmd, shell=True, stdout=PIPE, universal_newlines=True).communicate()
    outs = outs.strip().split('\n')[-1]
    assert outs.startswith("RESULT"), outs
    print(outs[len("RESULT: "):]+","+query_name)
