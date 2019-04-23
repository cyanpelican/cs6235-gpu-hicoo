
#ifndef CSF_HPP
#define CSF_HPP
#include "common.hpp"
#include "dense.hpp"
#include <memory>

struct CsfPoint {
    float value;
    unsigned int fiberX, fiberY; // the other 2 coordinates not related to the fiber
};


class DenseMatrix;
struct CsfTensor {
    CsfPoint* points_h;
    CsfPoint* points_d;
    unsigned int* fiberAddresses_h; // should be length (numFibers+1); fiber ends are stored implicitly
    unsigned int* fiberAddresses_d;
    unsigned int width, height, depth;
    unsigned int mode;

    CsfTensor() {
        points_h = nullptr;
        points_d = nullptr;
        fiberAddresses_h = nullptr;
        fiberAddresses_d = nullptr;
        width = 0;
        height = 0;
        depth = 0;
        mode = 0;
    }
    ~CsfTensor() {
        // handled by an owner
    }


    /* utility functions */

    // dangerous; deletes everything
    void freeAllArrays();

    // safely uploads to gpu
    void uploadToDevice();

    // safely downloads from gpu
    void downloadToHost();

    unsigned int numPoints() {
        return fiberAddresses_h[numFibers()];
    }
    unsigned int numFibers() {
        if(mode == 0) {
            return height * depth;
        } else if(mode == 1) {
            return width * depth;
        } else {
            return width * height;
        }
    }

    // // a safe function to get an element on either host or device; TODO
    // __device__ float& access() {
    //     return values_d[];
    // }
    //
    // __host__ float& access() {
    //     return values_h[];
    // }


    // void setSize() {
    //   // TODO
    // }
    
    //unsigned long long getTotalMemory() {
    //    return sizeof(CooPoint) * numPoints + sizeof(CooTensor);
    //}


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
struct CsfTensorUnique {
    CsfTensor tensor;

    CsfTensorUnique() {
        // nothing exciting to do
    }
    ~CsfTensorUnique() {
        tensor.freeAllArrays();
    }
};

// NOTE - build these, not CsfTensors; this does memory management
// However, when performing compute, just pass CsfTensors, since they're lighter.
// The operator() is overloaded, so it's possible to also use/pass these as if they're CsfTensors
struct CsfTensorManager {
    std::shared_ptr<CsfTensorUnique> tensor;

    CsfTensorManager():
      tensor(new CsfTensorUnique())
    {
    }

    /* utility functions */

    operator CsfTensor() {
        return tensor->tensor;
    }


    /* parsing, conversion & creation functions */
    // TODO

};

// Note - to me, the "fiber" in csf makes me feel like x and y should be implicit and z should be explicit,
// rather than having x be implicit and y/z be explicit. But who knows.

#endif
