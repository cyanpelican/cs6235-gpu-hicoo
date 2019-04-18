
#ifndef COO_HPP
#define COO_HPP
#include "common.hpp"
#include "dense.hpp"
#include <memory>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>

struct CooPoint {
    unsigned int x, y, z;
    unsigned int UNUSED; // for packing
    float value;
};

enum PointSorting {
    UNSORTED,
    XYZ,
    Z_MORTON
};

class DenseMatrix;
class HicooTensorManager;
class DenseTensorManager;
class CsfTensorManager;
struct CooTensor {
    CooPoint* points_h;
    CooPoint* points_d;
    PointSorting sorting;
    unsigned long long numElements;
    unsigned int width, height, depth;

    CooTensor() {
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

    // safely uploads to gpu
    void uploadToDevice();

    // safely downloads from gpu
    void downloadToHost();

    // a safe function to get an element on either host or device; TODO - test
    CooPoint& access(unsigned int element) {
        #ifdef __CUDA_ARCH__
            return points_d[element];
        #else
            return points_h[element];
        #endif
    }

    void setSize(int numPoints, int width, int height, int depth) {
        freeAllArrays();
        points_h = (CooPoint*)malloc(sizeof(CooPoint) * numPoints);
        this->numElements = numPoints;
    }
    
    unsigned long long getTotalMemory() {
        return sizeof(CooPoint) * numElements + sizeof(CooTensor);
    }


    /* conversion functions */
    HicooTensorManager toHicoo(int blockWidth = 2, int blockHeight = 2, int blockDepth = 2);
    DenseTensorManager toDense();
    CsfTensorManager   toCsf();


    /* compute functions */
    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    DenseMatrixManager mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c);
    DenseMatrixManager mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c);
    DenseMatrixManager mttkrp_fast(DenseMatrix d, DenseMatrix c);
    // TODO
};



// Don't make these. They're just middlemen.
struct CooTensorUnique {
    CooTensor tensor;

    CooTensorUnique() {
        // nothing exciting to do
    }
    ~CooTensorUnique() {
        tensor.freeAllArrays();
    }
};

// NOTE - build these, not CooTensors; this does memory management
// However, when performing compute, just pass CooTensors, since they're lighter.
// The operator() is overloaded, so it's possible to also use/pass these as if they're CooTensors
struct CooTensorManager {
    std::shared_ptr<CooTensorUnique> tensor;

    CooTensorManager():
      tensor(new CooTensorUnique()) {
    }

    /* utility functions */

    operator CooTensor() {
        return tensor->tensor;
    }


    /* parsing, conversion & creation functions */
    // TODO
    void create(char *tensorFileName);
    std::vector<double> split(const std::string *str, char delimiter);

};

#endif
