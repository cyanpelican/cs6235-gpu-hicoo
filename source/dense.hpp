
#ifndef DENSE_HPP
#define DENSE_HPP
#include "common.hpp"
#include <memory>

class DenseMatrix;
class DenseMatrixManager;
class CooTensorManager;
struct DenseTensor {
    float* values_h;
    float* values_d;
    unsigned int width, height, depth;

    DenseTensor() {
        values_h = nullptr;
        values_d = nullptr;
        width = 0;
        height = 0;
        depth = 0;
    }
    ~DenseTensor() {
        // handled by an owner
    }


    /* utility functions */

    // dangerous; deletes everything
    void freeAllArrays();

    // safely uploads to gpu
    void uploadToDevice();

    // safely downloads from gpu
    void downloadToHost();

    // a safe function to get an element on either host or device; TODO - test
    float& access(unsigned int i, unsigned int j, unsigned int k) {
        #ifdef __CUDA_ARCH__
            return values_d[i*height*width + j*width + k];
        #else
            return values_h[i*height*width + j*width + k];
        #endif
    }

    void setSize(unsigned int width, unsigned int height, unsigned int depth) {
        freeAllArrays();
        values_h = (float*)malloc(sizeof(float) * width*height*depth);
        this->width = width;
        this->height = height;
        this->depth = depth;
    }
    
    unsigned long long getTotalMemory() {
        return sizeof(float) * width*height*depth + sizeof(DenseTensor);
    }


    /* conversion function */
    CooTensorManager toCoo(float epsilon = 1e-4);


    /* compute functions */
    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    DenseMatrixManager mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c);
    DenseMatrixManager mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c);
    DenseMatrixManager mttkrp_fast(DenseMatrix d, DenseMatrix c);
    // TODO
};




struct DenseMatrix {
    float* values_h;
    float* values_d;
    unsigned int width, height;

    DenseMatrix() {
        values_h = nullptr;
        values_d = nullptr;
        width = 0;
        height = 0;
    }
    ~DenseMatrix() {
        // handled by an owner
    }


    /* utility functions */

    // dangerous; deletes everything
    void freeAllArrays();

    // safely uploads to gpu
    void uploadToDevice();

    // safely downloads from gpu
    void downloadToHost();

    // a safe function to get an element on either host or device; TODO - test
    float& access(unsigned int i, unsigned int j) {
        #ifdef __CUDA_ARCH__
            return values_d[i*width + j];
        #else
            return values_h[i*width + j];
        #endif
    }

    void setSize(unsigned int width, unsigned int height) {
        freeAllArrays();
        values_h = (float*)malloc(sizeof(float) * width*height);
        this->values_d = nullptr;
        this->width = width;
        this->height = height;
    }

    /* compute functions */
    // TODO
};










// Don't make these. They're just middlemen.
struct DenseTensorUnique {
    DenseTensor tensor;

    DenseTensorUnique() {
        // nothing exciting to do
    }
    ~DenseTensorUnique() {
        tensor.freeAllArrays();
    }
};

// NOTE - build these, not DenseTensors; this does memory management
// However, when performing compute, just pass DenseTensors, since they're lighter.
// The operator() is overloaded, so it's possible to also use/pass these as if they're DenseTensors
struct DenseTensorManager {
    std::shared_ptr<DenseTensorUnique> tensor;

    DenseTensorManager():
      tensor(new DenseTensorUnique())
    {
    }

    /* utility functions */

    operator DenseTensor() {
        return tensor->tensor;
    }


    /* parsing, conversion & creation functions */
    // TODO

};





// Don't make these. They're just middlemen.
struct DenseMatrixUnique {
    DenseMatrix tensor;

    DenseMatrixUnique() {
        // nothing exciting to do
    }
    ~DenseMatrixUnique() {
        tensor.freeAllArrays();
    }
};

// NOTE - build these, not DenseMatrixs; this does memory management
// However, when performing compute, just pass DenseMatrixs, since they're lighter.
// The operator() is overloaded, so it's possible to also use/pass these as if they're DenseMatrixs
struct DenseMatrixManager {
    std::shared_ptr<DenseMatrixUnique> tensor;

    DenseMatrixManager():
      tensor(new DenseMatrixUnique())
    {
    }

    /* utility functions */

    operator DenseMatrix() {
        return tensor->tensor;
    }


    /* parsing, conversion & creation functions */
    // TODO

};

#endif
