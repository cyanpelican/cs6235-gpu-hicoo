
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
class CooMatrixManager;
struct HicooTensor {
    HicooPoint* points_h;
    HicooPoint* points_d;
    HicooBlock* blocks_h;
    HicooBlock* blocks_d; // REMEMBER - there is one extra block to point to the end of points.
    PointSorting sorting;
    unsigned long long numPoints;
    unsigned long long numBlocks;
    // TODO - other things like sizes

    HicooTensor() {
        points_h = nullptr;
        points_d = nullptr;
        blocks_h = nullptr;
        blocks_d = nullptr;
        sorting = UNSORTED;
        numPoints = 0;
        numBlocks = 0;
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

    // a safe function to get a block on either host or device; TODO - test
    HicooBlock& access_block(unsigned int blockIndex) {
        #ifdef __CUDA_ARCH__
            return blocks_d[blockIndex];
        #else
            return blocks_h[blockIndex];
        #endif
    }

    // a safe function to get an element on either host or device; TODO - test
    HicooPoint& access_point(unsigned long long pointIndex) {
        #ifdef __CUDA_ARCH__
            return points_d[pointIndex];
        #else
            return points_h[pointIndex];
        #endif
    }


    void setSize(unsigned int numBlocks, unsigned int numPoints) {
        freeAllArrays();
        points_h = (HicooPoint*)malloc(sizeof(HicooPoint) * numPoints);
        blocks_h = (HicooBlock*)malloc(sizeof(HicooBlock) * numBlocks);
        this->numBlocks = numPoints;
        this->numPoints = numPoints;
    }
    
    unsigned long long getTotalMemory() {
        return sizeof(HicooPoint) * numPoints + sizeof(HicooBlock) * numBlocks + sizeof(HicooTensor);
    }


    /* conversion function */
    CooTensorManager toCoo();


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
