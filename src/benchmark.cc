#include "util.h"
#include "searches/branching_binary_search.h"
#include "competitors/rmi_search.h"
#include "competitors/rmi_universal.h"

#include <immintrin.h>
#include <sstream>
#include <algorithm>

using KeyType=uint64_t;
BranchingBinarySearch<KeyType> searcher;
bool cold_cache;
std::vector<Row<KeyType>> data_;
bool unique_keys_;
std::vector<KeyValue<KeyType>> index_data_;
std::vector<LBLookup<KeyType>> lookups_;
std::string data_filename_, lookups_filename_, prebuild_filename_;
size_t num_repeats_;
// Run times.
std::vector<uint64_t> runs_;
uint64_t random_sum, individual_ns_sum, build_ns_;

bool CheckResults(uint64_t actual, uint64_t expected,
                  KeyType lookup_key, SearchBound bound) {
  if (actual!=expected) {
    // const auto pos = std::find_if(
    //   data_.begin(),
    //   data_.end(),
    //   [lookup_key](const auto& kv) { return kv.key==lookup_key; }
    //   );
    
    // const auto idx = std::distance(data_.begin(), pos);
    auto idx = expected; // instead of returning sum, return the idx
    
    std::cerr << "equality lookup returned wrong result:" << std::endl;
    std::cerr << "lookup key: " << lookup_key << std::endl;
    std::cerr << "actual: " << actual << ", expected: " << expected
              << std::endl
              << "correct index is: " << idx << std::endl
              << "index start: " << bound.start << " stop: "
              << bound.stop << std::endl;
    
    return false;
  }

  return true;
}

template<class Index, bool clear_cache>
void DoLBLookups(Index& index) {
  size_t repeats = num_repeats_;
  size_t cntr(0);
  bool run_failed = false;
  
  std::vector<uint64_t> memory(26e6 / 8); // NOTE: L3 size of the machine
  if (clear_cache) {
    util::FastRandom ranny(8128);
    for(uint64_t& iter : memory) {
      iter = ranny.RandUint32();
    }
  }

  // Define function that contains lookup code.
  auto f = [&]() {
  while (true) {
    static constexpr std::size_t batch_size = 1u << 16;
    const size_t begin = cntr += batch_size;
    if (begin >= lookups_.size()) break;
    for (unsigned int idx = begin; idx < begin + batch_size && idx < lookups_.size();
          ++idx) {
      // Compute the actual index for debugging.
      const uint64_t lookup_key = lookups_[idx].key;
      const uint64_t expected = lookups_[idx].result;
      
      SearchBound bound;
      uint64_t actual;
      size_t qualifying;

      if (clear_cache) {
        // Make sure that all cache lines from large buffer are loaded
        for(uint64_t& iter : memory) {
          random_sum += iter;
        }
        _mm_mfence();

        const auto start = std::chrono::high_resolution_clock::now();
        bound = index.LBLookup(lookup_key);
        actual = searcher.search(
          data_, lookup_key,
          &qualifying,
          bound.start, bound.stop);
        if (!CheckResults(actual, expected, lookup_key, bound)) {
          run_failed = true;
          return;
        }
        const auto end = std::chrono::high_resolution_clock::now();
      
        const auto timing = std::chrono::duration_cast<std::chrono::nanoseconds>(
          end - start).count();
        individual_ns_sum += timing;
        
      } else {
        // not tracking errors, measure the lookup time.
        bound = index.LBLookup(lookup_key);
        actual = searcher.search(
          data_, lookup_key,
          &qualifying,
          bound.start, bound.stop);
        if (!CheckResults(actual, expected, lookup_key, bound)) {
          run_failed = true;
          return;
        }
      }
    }
  }
  };

  if (clear_cache)
    std::cout << "rsum was: " << random_sum << std::endl;

  runs_.resize(repeats);
  for (unsigned int i = 0; i < repeats; ++i) {
    random_sum = 0;
    individual_ns_sum = 0;
    cntr = 0;
    const auto ms = util::timing(f);
    // log_sum_search_bound_ /= static_cast<double>(lookups_.size());
    // l1_sum_search_bound_ /= static_cast<double>(lookups_.size());
    // l2_sum_search_bound_ /= static_cast<double>(lookups_.size());
    runs_[i] = ms;
    if (run_failed) {
      runs_ = std::vector<uint64_t>(repeats, 0);
      return;
    }
  }
}

template<class Index>
void PrintResult(const Index& index) {
  if (cold_cache) {
    double ns_per = ((double)individual_ns_sum) / ((double)lookups_.size());
    std::cout << "RESULT: " << index.name()
              << "," << index.variant()
              << "," << ns_per
              << "," << index.size() << "," << build_ns_
              << "," << searcher.name()
              << std::endl;
    return;
  }
  
  // print main results
  std::ostringstream all_times;
  for (unsigned int i = 0; i < runs_.size(); ++i) {
    const double ns_per_lookup = static_cast<double>(runs_[i])
      /lookups_.size();
    all_times << "," << ns_per_lookup;
  }
  // don't print a line if (the first) run failed
  if (runs_[0]!=0) {
    std::cout << "RESULT: " << index.name()
              << "," << index.variant()
              << all_times.str() // has a leading comma
              << "," << index.size() << "," << build_ns_
              << "," << searcher.name()
              << std::endl;
  }
}

template<class Index>
void Run() {
  // Build index.
  Index index;

  if (!index.applicable(unique_keys_, data_filename_)) {
    std::cout << "index " << index.name() << " is not applicable"
              << std::endl;
    return;
  }

  build_ns_ = index.Build(index_data_, prebuild_filename_);
  
  // Do equality lookups.
  if (cold_cache) {
    DoLBLookups<Index, true>(index);
    PrintResult(index);
  } else {
    DoLBLookups<Index, false>(index);
    PrintResult(index);
  }
}

void init(std::string data_filename, std::string lookups_filename,
  std::string prebuild_filename)
{
  cold_cache = false;
  num_repeats_ = 1;
  data_filename_ = data_filename;
  lookups_filename_ = lookups_filename;
  prebuild_filename_ = prebuild_filename;
  // Load data.
  std::vector<KeyType> keys = util::load_data<KeyType>(data_filename_);

  if (!std::is_sorted(keys.begin(), keys.end()))
    util::fail("keys have to be sorted");
  // Check whether keys are unique.
  unique_keys_ = util::is_unique(keys);
  if (unique_keys_)
    std::cout << "data is unique" << std::endl;
  else
    std::cout << "data contains duplicates" << std::endl;
  // Add artificial values to keys.
  data_ = util::add_values(keys);
  // Load lookups.
  lookups_ = util::load_data<LBLookup<KeyType>>(lookups_filename_);

  // Create the data for the index (key -> position).
  for (uint64_t lb_pos = 0, pos = 0; pos < data_.size(); pos++) {
    if (pos == 0 || data_[pos].key != data_[pos - 1].key) lb_pos = pos; // a new key, so update pos
    index_data_.push_back((KeyValue<KeyType>) {data_[pos].key, lb_pos}); // what's the use???
  }
}

int main(int argc, char* argv[])
{
  // usage: main <data_path> <lookups_path> [<prebuild_path>]
  auto data_filename_ = argv[1];
  auto lookups_filename_ = argv[2];
  auto prebuild_filename = "";
  if (argc > 3) prebuild_filename = argv[3];
  init(data_filename_, lookups_filename_, prebuild_filename);
  Run<RMI<uint64_t, rmi_universal::lookup, rmi_universal::load, rmi_universal::cleanup>>();
}