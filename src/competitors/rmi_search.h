#pragma once

#include "../util.h"
// #include "base.h"
// #include "rmi/all_rmis.h"

#include <cmath>
#include <algorithm>

//#define DEBUG_RMI
std::ifstream::pos_type filesize(const char* filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg(); 
}
std::string extract_variant(std::string rmi_path)
{
  std::string underscore = "_";
  auto lpos = std::find_end(rmi_path.begin(), rmi_path.end(), underscore.begin(), underscore.end());
  auto l2pos = std::find_end(rmi_path.begin(), lpos, underscore.begin(), underscore.end());
  if (lpos == rmi_path.end() || l2pos == rmi_path.end()) return "unknown";
  return std::string(l2pos + 1, lpos);
}
// RMI with binary search
template<class KeyType,
         uint64_t (* RMI_FUNC)(uint64_t, size_t*),
         bool (* RMI_LOAD)(char const*),
         void (* RMI_CLEANUP)()>
class RMI {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
    const std::string &prebuild_filename) {
    const std::string rmi_path = (std::getenv("SOSD_RMI_PATH") == NULL ?
                                  prebuild_filename : std::getenv("SOSD_RMI_PATH"));
    data_size_ = data.size();
    binary_size_ = filesize(rmi_path.c_str());
    variant_ = extract_variant(rmi_path);

    if (!RMI_LOAD(rmi_path.c_str())) {
      util::fail("Could not load RMI data from rmi_data/ -- either an allocation failed or the file could not be read.");
    }

    return 0;
  }

  SearchBound LBLookup(const KeyType lookup_key) const {
    size_t error;
    uint64_t guess = RMI_FUNC(lookup_key, &error);

    uint64_t start = (guess < error ? 0 : guess - error);
    uint64_t stop = (guess + error >= data_size_ ? data_size_ : guess + error);

    return (SearchBound){ start, stop };
  }

  std::string name() const {
    return "RMI";
  }

  std::size_t size() const {
    return binary_size_;
  }

  bool applicable(bool _unique, const std::string& data_filename) const {
    return true;
  }

  std::string variant() const { return variant_; }
  
  ~RMI() {
    RMI_CLEANUP();
  }
  
 private:
  uint64_t data_size_;
  uint64_t binary_size_;
  std::string variant_;
};
