set -e
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8
cd .. 

function benchmark() {

  dataset=$3
  out=./run/result_$2_${dataset}.txt
  [ -f ${out} ] || (
    python -u $1 ${dataset} | tee ./run/tmp.txt
    mv ./run/tmp.txt ./run/result_$2_${dataset}.txt
  )
}
benchmark ./scripts/benchmark_stxbtree.py stxbtree uniform_dense_200M_uint64
benchmark ./scripts/benchmark_stxbtree.py stxbtree wiki_ts_200M_uint64
benchmark ./scripts/benchmark.py baseline uniform_dense_200M_uint64
benchmark ./scripts/benchmark.py baseline wiki_ts_200M_uint64
