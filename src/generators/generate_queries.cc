#include "../util.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include "zipf.h"

using namespace std;
const int SEED = 97;
template<class KeyType>
vector<LBLookup<KeyType>> draw_n(const vector<KeyType> &keys, bool in_index,
 string dist_name, double alpha, size_t num_lookups)
{
  assert(keys.size() < (1ull<<31));
  vector<LBLookup<KeyType>> res(num_lookups);
  // Required to generate negative lookups within data domain.
  const KeyType min_key = keys.front();
  const KeyType max_key = keys.back();
  cout << "min_key=" << min_key << ", max_key=" << max_key << endl;
  if (in_index && dist_name == "uniform") {
    // vector<KeyType> unique_keys = util::remove_duplicates(keys);
    // std::mt19937 gen(SEED);

    //   vector<Row<KeyType>> data = util::add_values(keys);
    // std::uniform_int_distribution<int32_t> distrib(0, unique_keys.size() - 1);
    // util::FastRandom ranny(42);
    // for (size_t i = 0; i < num_lookups; ++i) {
    //   while (1) {
    //     KeyType key = unique_keys[ranny.RandUint32(0, unique_keys.size() - 1)]; //distrib(gen)];
    //     uint64_t lb = std::lower_bound(keys.begin(), keys.end(), key) - keys.begin();
    //     uint64_t cnt= 0, s = 0;
    //     for (auto j=lb; j < keys.size() && keys[j] == keys[lb]; ++j) ++cnt, s +=data[j].data[0];
    //     if (cnt > 100) continue;
    //     res[i] = {key, lb};
    //     break;
    //   }
    // }

    std::mt19937 gen(SEED);
    std::uniform_int_distribution<int32_t> distrib(0, keys.size() - 1);
    for (size_t i = 0; i < num_lookups; ++i) {
      uint64_t key = keys[distrib(gen)];
      uint64_t lb = std::lower_bound(keys.begin(), keys.end(), key) - keys.begin();
      res[i] = {key, lb};
    }
  } else if (in_index && dist_name == "zipf") {
    assert(false); // TODO
    // std::mt19937 gen(SEED);
    // ZipfDist distrib(alpha, keys.size());
    // for (size_t i = 0; i < num_lookups; ++i) {
    //   uint64_t key = keys[distrib(gen)];
    //   uint64_t lb = std::lower_bound(keys.begin(), keys.end(), key) - keys.begin();
    //   res[i] = {key, lb};
    // }
    // break;
  } else if (!in_index && dist_name == "uniform") {
    std::mt19937_64 gen(SEED);
    std::uniform_int_distribution<uint64_t> distrib(min_key, max_key);
    for (size_t i = 0; i < num_lookups; ++i) {
      uint64_t key = distrib(gen);
      uint64_t lb = std::lower_bound(keys.begin(), keys.end(), key) - keys.begin();
      res[i] = {key, lb};
    }
  } else if (!in_index && dist_name == "zipf") {
    assert(false);
  } else {
    cerr << "Unrecognized argument!" << endl;
    assert(false);
  }
  // validate
  for (auto&& [k,v] : res) {
    assert(k >= min_key && k <= max_key);
    assert(v >= 0 && v < keys.size());
  }
  return res;
}
int main(int argc, char* argv[]) {
  if (argc < 6)
    util::fail(
        "usage: ./generate <data file> <num lookups> <output file> <uniform|zipf> <alpha> [in_index]");

  const string filename = argv[1];
  const DataType type = util::resolve_type(filename);
  size_t num_lookups = stoull(argv[2]);
  const string output_path = argv[3];
  const string dist_name = argv[4];
  assert(dist_name == "uniform" || dist_name == "zipf");
  const double alpha = stod(argv[5]);
  bool in_index = (argc == 7);
  if (argc == 7) assert(string(argv[6]) == "in_index");
  cout << "using " << dist_name << ", in_index=" << in_index << endl;

  switch (type) {
    case DataType::UINT32: {
      assert(false);
    }
    case DataType::UINT64: {
      using KeyType = uint64_t;
      // Load data.
      const vector<KeyType> keys = util::load_data<KeyType>(filename);
      if (!is_sorted(keys.begin(), keys.end()))
        util::fail("keys have to be sorted (read 64-bit keys)");
      cout << "number of unique keys=" << util::remove_duplicates(keys).size() << endl;
      // Generate benchmarks.
      auto lookups = draw_n(keys, in_index, dist_name, alpha, num_lookups);
      util::write_data(lookups, output_path);
      break;
    }
  }

  return 0;
}
