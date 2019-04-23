
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
    unsigned int width, height, depth;
    unsigned int blockWidth, blockHeight, blockDepth;

    HicooTensor() {
        DEBUG_PRINT("HT: constructor\n");
        points_h = nullptr;
        points_d = nullptr;
        blocks_h = nullptr;
        blocks_d = nullptr;
        sorting = UNSORTED;
        numPoints = 0;
        numBlocks = 0;

        width = 0;
        height = 0;
        depth = 0;
        blockWidth = 0;
        blockHeight = 0;
        blockDepth = 0;
    }
    ~HicooTensor() {
        // handled by an owner
    }


    /* utility functions */

    // dangerous; deletes everything
    void freeAllArrays();
    void freeHostArrays();
    void freeDeviceArrays();

    // safely uploads to gpu
    void uploadToDevice();

    // safely downloads from gpu
    void downloadToHost();

    // a safe function to get a block on either host or device; TODO - test
    HicooBlock& __host__ __device__ access_block(unsigned int blockIndex) {
        #ifdef __CUDA_ARCH__
            return blocks_d[blockIndex];
        #else
            return blocks_h[blockIndex];
        #endif
    }

    // a safe function to get an element on either host or device; TODO - test
    HicooPoint& __host__ __device__ access_point(unsigned long long pointIndex) {
        #ifdef __CUDA_ARCH__
            return points_d[pointIndex];
        #else
            return points_h[pointIndex];
        #endif
    }

    // a safe function to get an element on either host or device; TODO - test
    HicooPoint& __host__ __device__ access_pointInBlock(unsigned int blockIndex, unsigned long long pointIndex) {
        return access_point(pointIndex + access_block(blockIndex).blockAddress);
    }


    void setSize(unsigned int numBlocks, unsigned int numPoints, unsigned int width, unsigned int height, unsigned int depth, unsigned int blockWidth, unsigned int blockHeight, unsigned int blockDepth) {
        DEBUG_PRINT("HT: set size (nb %d, np %d, w %d, h %d, d %d, bw %d, bh %d, bd %d)\n", numBlocks, numPoints, width, height, depth, blockWidth, blockHeight, blockDepth);
        freeAllArrays();
        points_h = (HicooPoint*)malloc(sizeof(HicooPoint) * numPoints);
        blocks_h = (HicooBlock*)malloc(sizeof(HicooBlock) * (numBlocks+1));
        assert(points_h != nullptr);
        assert(blocks_h != nullptr);
        this->numBlocks = numBlocks;
        this->numPoints = numPoints;

        this->width = width;
        this->height = height;
        this->depth = depth;
        this->blockWidth = blockWidth;
        this->blockHeight = blockHeight;
        this->blockDepth = blockDepth;
    }

    unsigned long long getTotalMemory() {
        DEBUG_PRINT("HT: get total memory\n");
        return sizeof(HicooPoint) * numPoints + sizeof(HicooBlock) * (numBlocks+1) + sizeof(HicooTensor);
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
        DEBUG_PRINT("HTU: auto-free from unique destructor\n");
        tensor.freeAllArrays();
    }
};

// NOTE - build these, not HicooTensors; this does memory management
// However, when performing compute, just pass HicooTensors, since they're lighter.
// The cast operator is overloaded, so it's possible to also use/pass these as if they're HicooTensors
// MAKE SURE that when you cast it, you:
//  - keep it as a HicooTensor& [don't forget the ampersand] if you're going to modify any properties or alloc/free pointers
//  - pass it as a HicooTensor  [no ampersand] when passing to the GPU
struct HicooTensorManager {
    std::shared_ptr<HicooTensorUnique> tensor;

    HicooTensorManager():
      tensor(new HicooTensorUnique())
    {
        DEBUG_PRINT("HTM: constructor\n");
    }

    /* utility functions */

    operator HicooTensor&() {
        return tensor->tensor;
    }


    /* parsing, conversion & creation functions */
    // TODO

};

#endif
