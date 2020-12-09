python ./scripts/build_baseline_rmi.py --param-grid ./scripts/rmi_specs/wiki_ts_200M_uint64.json --data ./data/wiki_ts_200M_uint64 --output-path ./rmi_data/baseline --RMI-path $1
python ./scripts/build_baseline_rmi.py --param-grid ./scripts/rmi_specs/uniform_dense_200M_uint64.json --data ./data/uniform_dense_200M_uint64 --output-path ./rmi_data/baseline --RMI-path $1
