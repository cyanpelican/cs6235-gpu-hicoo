
#ifndef HICOO_HPP
#define HICOO_HPP
#include "coo.hpp"

struct HicooPoint {
    unsigned char x, y, z;
    unsigned char UNUSED; // for packing
    float value;
};

struct HicooBlock {
    unsigned long long blockAddress;
    unsigned int blockX, blockY, blockZ;
    unsigned int UNUSED; // for packing
}

struct COOTensor {
    HicooPoint* points_h;
    HicooPoint* points_d;
    HicooBlock* blocks_h;
    HicooBlock* blocks_d;
    PointSorting sorting;
    unsigned long long num_elements;
    unsigned long long num_blocks;
};

#endif
