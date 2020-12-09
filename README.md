Course project for FML. Code mainly adapted from <https://github.com/learnedsystems/SOSD>.

### Step 1: Loading Data

Two options: 

1. To build from scratch:

```shell
bash ./scripts/generate.sh # generate data
# IMPORTANT: clone the https://github.com/learnedsystems/RMI repo and replace the path below
bash ./scripts/build_baseline_rmi.sh /path/to/the/RMI/repo
```

2. To download prebuild data and baseline:

```shell
bash ./scripts/download_prebuild_data.sh
```

### Step 2: Run Benchmark

To benchmark all the baselines:

```shell
bash ./scripts/benchmark_baseline.sh
```

Result will be saved to `run/result/` with each line of format as follows:

```
<model_name>,<variant_id>,<run_time_1>,<run_time_2>,<run_time_3>,<run_time_4>,<run_time_5>,<model_size>,<build_time>,<last_mile_searcher>,<workload>
```

For example:

```
RMI,3,1343.46,927.907,877.172,919.756,937.959,24080,0,BinarySearch,wiki_ts_200M_uint64_queries_10M_in_test
RMI,4,1190.39,649.651,669.04,648.392,668.077,240080,0,BinarySearch,wiki_ts_200M_uint64_queries_10M_in_test
RMI,5,1030.91,425.019,429.79,398.885,398.248,2400080,0,BinarySearch,wiki_ts_200M_uint64_queries_10M_in_test
RMI,6,559.205,371.127,352.677,352.18,344.074,24000080,0,BinarySearch,wiki_ts_200M_uint64_queries_10M_in_test
```

To test your own RMI model, please refer to `src/benchmark.cc`. For example:

```bash
./build/benchmark ./data/wiki_ts_200M_uint64 ./data/wiki_ts_200M_uint64_queries_10M_in_test 5 rmi /path/to/your/rmi/parameter/data
```

This will use query workload `./data/wiki_ts_200M_uint64_queries_10M_in_test` to benchmark the RMI model with parameters `/path/to/your/rmi/parameter/data`.

