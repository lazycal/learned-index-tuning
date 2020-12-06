#pragma once

#include "../util.h"
// #include "base.h"
// #include "rmi/all_rmis.h"

#include <math.h>

//#define DEBUG_RMI

// RMI with binary search
template<class KeyType,
         uint64_t (* RMI_FUNC)(uint64_t, size_t*),
         bool (* RMI_LOAD)(char const*),
         void (* RMI_CLEANUP)()>
class RMI {
 public:
  uint64_t Build(const std::vector<KeyValue<KeyType>>& data,
    const std::string &prebuild_filename) {
    data_size_ = data.size();;

    const std::string rmi_path = (std::getenv("SOSD_RMI_PATH") == NULL ?
                                  prebuild_filename : std::getenv("SOSD_RMI_PATH"));
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
    return 0;
  }

  bool applicable(bool _unique, const std::string& data_filename) const {
    return true;
  }

  int variant() const { return 0; }
  
  ~RMI() {
    RMI_CLEANUP();
  }
  
 private:
  uint64_t data_size_;
};
