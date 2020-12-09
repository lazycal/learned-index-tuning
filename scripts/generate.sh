#! /usr/bin/env bash
set -e

echo "Compiling..."
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8 

function generate_lookups() {
  echo "===> Generating uniform queries for $1 ..."
  path=../data/$1_queries_10M
  [ -f ${path}_in_train ] || ./generate_queries ../data/$1 10000000 ${path}_in uniform 0 in_index
  [ -f ${path}_out_train ] || ./generate_queries ../data/$1 10000000 ${path}_out uniform 0
  for alpha in 0.5 0.99; do
    echo "===> Generating zipf($alpha) queries for $1 ..."
    path=../data/$1_zipf${alpha}_queries_10M
    [ -f ${path}_in_train ] || ./generate_queries ../data/$1 10000000 ${path}_in zipf ${alpha} in_index
    [ -f ${path}_out_train ] || ./generate_queries ../data/$1 10000000 ${path}_out zipf ${alpha}
  done
}

python ./src/generators/gen_uniform.py --many
generate_lookups uniform_dense_200M_uint64
# generate_lookups osm_cellids_200M_uint64
generate_lookups wiki_ts_200M_uint64
# generate_lookups books_200M_uint32
# generate_lookups books_200M_uint64
# generate_lookups fb_200M_uint64
