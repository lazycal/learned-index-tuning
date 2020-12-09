set -e
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8
cd .. 

mkdir -p run/result
function benchmark() {

  dataset=$3
  out=./run/result/result_$2_${dataset}.txt
  [ -f ${out} ] || (
    python -u $1 ${dataset} | tee ./run/result/tmp.txt
    mv ./run/result/tmp.txt ${out}
  )
}
[ ! -f ./data/uniform_dense_200M_uint64 ] || benchmark ./scripts/benchmark_stxbtree.py stxbtree uniform_dense_200M_uint64
benchmark ./scripts/benchmark_stxbtree.py stxbtree wiki_ts_200M_uint64
[ ! -f ./data/uniform_dense_200M_uint64 ] || benchmark ./scripts/benchmark_basermi.py basermi uniform_dense_200M_uint64
benchmark ./scripts/benchmark_basermi.py basermi wiki_ts_200M_uint64
