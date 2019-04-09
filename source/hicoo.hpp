
#ifndef HICOO_HPP
#define HICOO_HPP
#include "common.hpp"
#include "coo.hpp"
#include "dense.hpp"
#include <memory>

struct HicooPoint {
    unsigned char x, y, z;
    unsigned char UNUSED; // for packing
    float value;
};

struct HicooBlock {
    unsigned long long blockAddress;
    unsigned int blockX, blockY, blockZ;
    unsigned int UNUSED; // for packing
};

class DenseMatrix;
struct HicooTensor {
    HicooPoint* points_h;
    HicooPoint* points_d;
    HicooBlock* blocks_h;
    HicooBlock* blocks_d;
    PointSorting sorting;
    unsigned long long num_elements;
    unsigned long long num_blocks;

    HicooTensor() {
        points_h = nullptr;
        points_d = nullptr;
        sorting = UNSORTED;
        num_elements = 0;
    }
    ~HicooTensor() {
        // handled by an owner
    }


    /* utility functions */

    // dangerous; deletes everything
    void freeAllArrays();

    // safely uploads to gpu
    void uploadToDevice();

    // safely downloads from gpu
    void downloadToHost();


    /* compute functions */
    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    DenseMatrixManager mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c);
    DenseMatrixManager mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c);
    DenseMatrixManager mttkrp_fast(DenseMatrix d, DenseMatrix c);
    // TODO
};



// Don't make these. They're just middlemen.
struct HicooTensorUnique {
    HicooTensor tensor;

    HicooTensorUnique() {
        // nothing exciting to do
    }
    ~HicooTensorUnique() {
        tensor.freeAllArrays();
    }
};

// NOTE - build these, not HicooTensors; this does memory management
// However, when performing compute, just pass HicooTensors, since they're lighter.
// The operator() is overloaded, so it's possible to also use/pass these as if they're HicooTensors
struct HicooTensorManager {
    std::shared_ptr<HicooTensorUnique> tensor;

    HicooTensorManager():
      tensor(new HicooTensorUnique())
    {
    }

    /* utility functions */

    operator HicooTensor() {
        return tensor->tensor;
    }


    /* parsing, conversion & creation functions */
    // TODO

};

#endif
