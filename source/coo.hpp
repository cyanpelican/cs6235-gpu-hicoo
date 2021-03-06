
#ifndef COO_HPP
#define COO_HPP

#include <memory>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <assert.h>
#include "common.hpp"
#include "dense.hpp"


struct CooPoint {
    unsigned int x, y, z;
    float value;
};

enum PointSorting {
    UNSORTED,
    XYZ,
    ZYX,
    Z_MORTON
};

class DenseMatrixManager;
class HicooTensorManager;
class DenseTensorManager;
struct CooTensor {
    CooPoint* points_h;
    CooPoint* points_d;
    PointSorting sorting;
    unsigned long long numElements;
    unsigned int width, height, depth;

    CooTensor() {
        DEBUG_PRINT("CT: constructor\n");
        points_h = nullptr;
        points_d = nullptr;
        sorting = UNSORTED;
        numElements = 0;
        width = 0;
        height = 0;
        depth = 0;
    }
    ~CooTensor() {
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
    CooPoint& __host__ __device__ access(unsigned int element) {
        #ifdef __CUDA_ARCH__
            return points_d[element];
        #else
            #if ENABLE_ACCESS_ASSERTS
              assert(element < numElements);
            #endif
            return points_h[element];
        #endif
    }


//    float __host__ __device__ access(int x, int y, int z) {
//        for (unsigned int i = 0; i < this->numElements; i++) {
//            if(access(i).x == x && access(i).y == y && access(i).z == z) {
//                #ifdef __CUDA_ARCH__
//                    return points_d[i].value;
//                #else
//                    return points_h[i].value;
//                #endif
//            }
//        }
//
//        //value not found: it's a 0
//        return 0.0;
//    }
//
//    __host__ __device__ float access_sorted(int x, int y, int z) {
//        for (unsigned int i = 0; i < this->numElements; i++) {
//            if (access(i).x == x && access(i).y == y && access(i).z == z) {
//                #ifdef __CUDA_ARCH__
//                    return points_d[i].value;
//                #else
//                    return points_h[i].value;
//                #endif
//            }
//            //or is it the z we should be checking...?
//            if (access(i).x > x)
//                break;
//        }
//
//        //value not found: it's a 0
//        return 0.0;
//    }

    void setSize(int numPoints, int depth, int height, int width) {
        DEBUG_PRINT("CT: setSize (# %d, d %d, h %d, w %d)\n", numPoints, depth, height, width);
        freeAllArrays();
        points_h = (CooPoint*)malloc(sizeof(CooPoint) * numPoints);
        assert(points_h != nullptr);
        this->numElements = numPoints;
        this->width = width;
        this->height = height;
        this->depth = depth;
    }

    unsigned long long getTotalMemory() {
        DEBUG_PRINT("CT: get total memory\n");
        return sizeof(CooPoint) * numElements + sizeof(CooTensor);
    }


    /* conversion functions */
    HicooTensorManager toHicoo(int blockWidth = 2, int blockHeight = 2, int blockDepth = 2);
    DenseTensorManager toDense();


    /* compute functions */
    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    DenseMatrixManager mttkrp_naive_cpu(DenseMatrixManager d, DenseMatrixManager c);
    DenseMatrixManager mttkrp_naive_gpu(DenseMatrixManager d, DenseMatrixManager c);

    DenseMatrixManager mttkrp_kevin1(DenseMatrixManager d, DenseMatrixManager c);
};



// Don't make these. They're just middlemen.
struct CooTensorUnique {
    CooTensor tensor;

    CooTensorUnique() {
        // nothing exciting to do
    }
    ~CooTensorUnique() {
        DEBUG_PRINT("CTU: auto-free from unique destructor\n");
        tensor.freeAllArrays();
    }
};

// NOTE - build these, not CooTensors; this does memory management
// However, when performing compute, just pass CooTensors, since they're lighter.
// The cast operator is overloaded, so it's possible to also use/pass these as if they're CooTensors
// MAKE SURE that when you cast it, you:
//  - keep it as a CooTensor& [don't forget the ampersand] if you're going to modify any properties or alloc/free pointers
//  - pass it as a CooTensor  [no ampersand] when passing to the GPU
struct CooTensorManager {
    std::shared_ptr<CooTensorUnique> tensor;

    CooTensorManager():
      tensor(new CooTensorUnique()) {
        DEBUG_PRINT("CTM: constructor\n");
    }

    /* utility functions */

    operator CooTensor&() {
        return tensor->tensor;
    }


    /* parsing, conversion & creation functions */
    // TODO
    void create(char *tensorFileName);

};

#endif
