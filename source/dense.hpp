
#ifndef DENSE_HPP
#define DENSE_HPP
#include "common.hpp"
#include <memory>
#include <assert.h>

class DenseMatrix;
class DenseMatrixManager;
class CooTensorManager;
struct DenseTensor {
    float* values_h;
    float* values_d;
    unsigned int width, height, depth;

    DenseTensor() {
        DEBUG_PRINT("DT: constructor\n");
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
    void freeHostArrays();
    void freeDeviceArrays();

    // safely uploads to gpu
    void uploadToDevice();

    // safely downloads from gpu
    void downloadToHost();

    // a handy function to get an element on either host or device
    float& __host__ __device__ access(unsigned int i, unsigned int j, unsigned int k) {
        #ifdef __CUDA_ARCH__
            return values_d[i*height*width + j*width + k];
        #else
            return values_h[i*height*width + j*width + k];
        #endif
    }

    void setSize(unsigned int depth, unsigned int height, unsigned int width) {
        DEBUG_PRINT("DT: setSize (d %d, h %d, w %d)\n", depth, height, width);
        freeAllArrays();
        values_h = (float*)malloc(sizeof(float) * width*height*depth);
        assert(values_h != nullptr);
        memset(values_h, 0.0f, width*height*depth * sizeof(float));
        this->width = width;
        this->height = height;
        this->depth = depth;
    }

    unsigned long long getTotalMemory() {
        DEBUG_PRINT("DT: get total memory\n");
        return sizeof(float) * width*height*depth + sizeof(DenseTensor);
    }


    /* conversion function */
    CooTensorManager toCoo(float epsilon = 1e-4);


    /* compute functions */
    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    DenseMatrixManager mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c);
    DenseMatrixManager mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c);
    DenseMatrixManager mttkrp_fast(DenseMatrix d, DenseMatrix c);
};




struct DenseMatrix {
    float* values_h;
    float* values_d;
    unsigned int width, height;

    DenseMatrix() {
        DEBUG_PRINT("DT: constructor\n");
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
    void freeHostArrays();
    void freeDeviceArrays();

    // safely uploads to gpu
    void uploadToDevice();

    // safely downloads from gpu
    void downloadToHost();

    // a handy function to get an element on either host or device
    float& __host__ __device__ access(unsigned int i, unsigned int j) {
        #ifdef __CUDA_ARCH__
            return values_d[i*width + j];
        #else
            return values_h[i*width + j];
        #endif
    }

    void setSize(unsigned int height, unsigned int width) {
        DEBUG_PRINT("DM: set size (h %d, w %d)\n", height, width);
        freeAllArrays();
        values_h = (float*)malloc(sizeof(float) * width*height);
        assert(values_h != nullptr);
        memset(values_h, 0.0f, width*height * sizeof(float));
        this->width = width;
        this->height = height;
    }
    void setSize_d(unsigned int height, unsigned int width);

    unsigned long long getTotalMemory() {
        DEBUG_PRINT("DM: get total memory\n");
        return sizeof(float) * width*height + sizeof(DenseMatrix);
    }

    // TODO
    //  - have a create function, even an all 1's dense matrix will do
};










// Don't make these. They're just middlemen.
struct DenseTensorUnique {
    DenseTensor tensor;

    DenseTensorUnique() {
        // nothing exciting to do
    }
    ~DenseTensorUnique() {
        DEBUG_PRINT("DTU: auto-free from unique destructor\n");
        tensor.freeAllArrays();
    }
};

// NOTE - build these, not DenseTensors; this does memory management
// However, when performing compute, just pass DenseTensors, since they're lighter.
// The cast operator is overloaded, so it's possible to also use/pass these as if they're DenseTensors
// MAKE SURE that when you cast it, you:
//  - keep it as a DenseTensor& [don't forget the ampersand] if you're going to modify any properties or alloc/free pointers
//  - pass it as a DenseTensor  [no ampersand] when passing to the GPU
struct DenseTensorManager {
    std::shared_ptr<DenseTensorUnique> tensor;

    DenseTensorManager():
      tensor(new DenseTensorUnique())
    {
        DEBUG_PRINT("DTM: constructor\n");
    }

    /* utility functions */

    operator DenseTensor&() {
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
        DEBUG_PRINT("DMU: auto-free from unique destructor\n");
        tensor.freeAllArrays();
    }
};

// NOTE - build these, not DenseMatrixes; this does memory management
// However, when performing compute, just pass DenseMatrixs, since they're lighter.
// The cast operator is overloaded, so it's possible to also use/pass these as if they're DenseMatrixes
// MAKE SURE that when you cast it, you:
//  - keep it as a DenseMatrix& [don't forget the ampersand] if you're going to modify any properties or alloc/free pointers
//  - pass it as a DenseMatrix  [no ampersand] when passing to the GPU
struct DenseMatrixManager {
    std::shared_ptr<DenseMatrixUnique> tensor;

    DenseMatrixManager():
      tensor(new DenseMatrixUnique())
    {
        DEBUG_PRINT("DMM: constructor\n");
    }

    /* utility functions */

    operator DenseMatrix&() {
        return tensor->tensor;
    }

};

#endif
