// from Changgeng
#pragma once

#include <pthread.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>

/**********************************************************************
 * zipf distribution
 *********************************************************************/

int64_t FNV_OFFSET_BASIS_64 = 0xCBF29CE484222325;
int64_t FNV_PRIME_64 = 1099511628211;

class ZipfianGenerator {
   public:
    ZipfianGenerator(uint64_t min, uint64_t max)
        : items(max - min + 1),
          base(min),
          zipfianconstant(0.99),
          theta(0.99),
          dis(0, 1) {
        zetan = zeta(0, max - min + 1, zipfianconstant, 0);
        zeta2theta = zeta(0, 2, theta, 0);
        alpha = 1.0 / (1.0 - theta);
        eta = (1 - pow(2.0 / items, 1 - theta)) / (1 - zeta2theta / zetan);
        countforzeta = items;
        // nextValue();
    }

    ZipfianGenerator(uint64_t min, uint64_t max, double thet)
        : items(max - min + 1),
          base(min),
          zipfianconstant(thet),
          theta(thet),
          dis(0, 1) {
        zetan = zeta(0, max - min + 1, zipfianconstant, 0);
        zeta2theta = zeta(0, 2, theta, 0);
        alpha = 1.0 / (1.0 - theta);
        eta = (1 - pow(2.0 / items, 1 - theta)) / (1 - zeta2theta / zetan);
        countforzeta = items;
        // nextValue();
    }

    ZipfianGenerator(uint64_t min, uint64_t max, double thet, double zet)
        : items(max - min + 1),
          base(min),
          zipfianconstant(thet),
          theta(thet),
          zetan(zet),
          dis(0, 1) {
        zeta2theta = zeta(0, 2, theta, 0);
        alpha = 1.0 / (1.0 - theta);
        eta = (1 - pow(2.0 / items, 1 - theta)) / (1 - zeta2theta / zetan);
        countforzeta = items;
        // nextValue();
    }

    ~ZipfianGenerator() { }

    inline double zeta(uint64_t st, uint64_t n, double theta, double initialsum) {
        countforzeta = n;
        double sum = initialsum;
        if (n - st > 1000'000'000) {
          std::cerr << "n-st=" << n-st << " is too large. Please consider precomputing."
            << std::endl;
          std::exit(1);
        }
        std::cout << "computing zeta(st=" << st << ", n=" << n << ", theta=" << theta
          << ", initialsum=" << initialsum << std::endl;
        for (size_t i = st; i < n; i++) {
            sum += 1 / (pow(i + 1, theta));
        }
        std::cout << "zeta=" << sum << std::endl;
        return sum;
    }

    template<class Generator>
    inline uint64_t nextValue(Generator &gen) {
        // from "Quickly Generating Billion-Record Synthetic Databases", Jim Gray
        // et al, SIGMOD 1994
        double u = dis(gen);
        double uz = u * zetan;
        if (uz < 1.0)
            return base;
        if (uz < 1.0 + pow(0.5, theta))
            return base + 1;
        uint64_t ret = base + (uint64_t)(items * pow(eta * u - eta + 1, alpha));
        return ret;
    }

    template<class Generator>
    uint64_t nextHashed(Generator &gen) {
        uint64_t ret = nextValue(gen);
        ret = base + fnvhash64(ret) % items;
        // LOG(2) << ret;
        return ret;
    }

    uint64_t fnvhash64(uint64_t val) {
        // from http://en.wikipedia.org/wiki/Fowler_Noll_Vo_hash
        uint64_t hashval = FNV_OFFSET_BASIS_64;
        for (size_t i = 0; i < 8; i++) {
            uint64_t octet = val & 0x00ff;
            val = val >> 8;
            hashval = hashval ^ octet;
            hashval = hashval * FNV_PRIME_64;
        }
        return hashval;
    }
   private:
    // Number of items.
    uint64_t items;
    // Min item to generate.
    uint64_t base;
    uint64_t countforzeta;
    // The zipfian constant to use.
    double zipfianconstant;
    // Computed parameters for generating the distribution.
    double theta, zetan, zeta2theta, alpha, eta;
    // std::random_device rd;
    // std::mt19937 gen;
    std::uniform_real_distribution<> dis;
    // constexpr static double ZETAN = 26.46902820178302;
    // constexpr static uint64_t ITEM_COUNT = 10'000'000'000ull;
};



