#include "rmi_universal.h"
// #include "wiki_ts_200M_uint64_9_data.h"
#include <math.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace rmi_universal {
// hack
double L0_PARAMETER0;// = 0.0;
double L0_PARAMETER1;// = 0.0;
double L0_PARAMETER2;// = 0.0709191103324514;
double L0_PARAMETER3;// = -69477474.67147279;
size_t L1_NUM_MODELS;// = 16777216;
char* L1_PARAMETERS;
const uint64_t VERSION_NUM = 0;
int L1_model_params(bool is_last) 
{ 
  return 2 + is_last;
}
bool load(char const* dataPath) {
  {
    std::ifstream infile(dataPath, std::ios::in | std::ios::binary);
    if (!infile.good()) return false;
    uint64_t ver, num_layers, t1, t2;
    infile.read((char*)&ver, sizeof(ver));
    if (ver != VERSION_NUM) {
      std::cerr << "Version mismatched." << "Expected " << VERSION_NUM 
        << ", got " << ver << std::endl;
      exit(EXIT_FAILURE);
    }
    infile.read((char*)&num_layers, sizeof(num_layers));
    if (num_layers != 2) {
      // TODO
      std::cerr << "2 layers only" << std::endl;
      exit(EXIT_FAILURE);
    }
    infile.read((char*)&t1, 8);
    infile.read((char*)&t2, 8);
    std::cout << "t1=" << t1 << ", t2=" << t2 << "\n";
    infile.read((char*)&t2, 8);
    if (t2 != 1) {std::cerr << "t2=" << t2 << "!=1" << std::endl; exit(EXIT_FAILURE);}
    infile.read((char*)&L1_NUM_MODELS, 8);
    std::cout << "L1_NUM_MODELS=" << L1_NUM_MODELS << std::endl;

    infile.read((char*)&L0_PARAMETER0, 8);
    infile.read((char*)&L0_PARAMETER1, 8);
    infile.read((char*)&L0_PARAMETER2, 8);
    infile.read((char*)&L0_PARAMETER3, 8);
    auto sz = L1_model_params(true)*L1_NUM_MODELS*8;
    L1_PARAMETERS = (char*) malloc(sz);
    if (L1_PARAMETERS == NULL) return false;
    infile.read((char*)L1_PARAMETERS, sz);//402653184);
    if (!infile.good()) return false;
  }
  return true;
}
void cleanup() {
    free(L1_PARAMETERS);
}

inline double cubic(double a, double b, double c, double d, double x) {
    auto v1 = std::fma(a, x, b);
    auto v2 = std::fma(v1, x, c);
    auto v3 = std::fma(v2, x, d);
    return v3;
}

inline double linear(double alpha, double beta, double inp) {
    return std::fma(beta, inp, alpha);
}

inline size_t FCLAMP(double inp, double bound) {
  if (inp < 0.0) return 0;
  return (inp > bound ? bound : (size_t)inp);
}

uint64_t lookup(uint64_t key, size_t* err) {
  size_t modelIndex;
  double fpred;
  fpred = cubic(L0_PARAMETER0, L0_PARAMETER1, L0_PARAMETER2, L0_PARAMETER3, (double)key);
  modelIndex = std::min((size_t) std::max(0., fpred), L1_NUM_MODELS - 1);
  fpred = linear(*((double*) (L1_PARAMETERS + (modelIndex * 24) + 0)), *((double*) (L1_PARAMETERS + (modelIndex * 24) + 8)), (double)key);
  *err = *((uint64_t*) (L1_PARAMETERS + (modelIndex * 24) + 16));

  return FCLAMP(fpred, 200000000.0 - 1.0);
}
} // namespace
