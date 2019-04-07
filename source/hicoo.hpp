
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
    void freeAllArrays() {
        free(points_h);
        cudaErrorCheck(cudaFree(tensor.points_d));
    }

    // safely uploads to gpu
    void uploadToDevice() {
        cudaErrorCheck(cudaFree(tensor.points_d));
        cudaErrorCheck(cudaMalloc((void **) &d_weight, sizeof(float)));
        cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(HicooPoint) * num_elements, cudaMemcpyHostToDevice));
    }

    // safely downloads from gpu
    void downloadToHost() {
        free(points_h);
        points_h = malloc(sizeof(HicooPoint) * num_elements);
        cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(HicooPoint) * num_elements, cudaMemcpyDeviceToHost));
    }


    /* compute functions */
    // TODO
};



// Don't make these. They're just middlemen.
struct HicooTensorUnique {
    HicooTensor tensor;

    HicooTensorUnique() {
        // nothing exciting to do
    }
    ~HicooTensorUnique() {
        tensor.freeAllArrays()
    }
};

// NOTE - build these, not HicooTensors; this does memory management
// However, when performing compute, just pass HicooTensors, since they're lighter.
// The operator() is overloaded, so it's possible to also use/pass these as if they're HicooTensors
struct HicooTensorManager {
    std::shared_ptr<HicooTensorUnique> tensor(new HicooTensorUnique());

    /* utility functions */

    HicooTensor operator() {
        return tensor->tensor;
    }


    /* parsing, conversion & creation functions */
    // TODO

};

#endif
