
#ifndef COO_HPP
#define COO_HPP
#include "dense.hpp"

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
struct CooTensor {
    CooPoint* points_h;
    CooPoint* points_d;
    PointSorting sorting;
    unsigned long long num_elements;

    CooTensor() {
        points_h = nullptr;
        points_d = nullptr;
        sorting = UNSORTED;
        num_elements = 0;
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


    /* compute functions */
    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    DenseMatrixManager mttkrp(DenseMatrix d, DenseMatrix c);
    // TODO
};



// Don't make these. They're just middlemen.
struct CooTensorUnique {
    CooTensor tensor;

    CooTensorUnique() {
        // nothing exciting to do
    }
    ~CooTensorUnique() {
        tensor.freeAllArrays()
    }
};

// NOTE - build these, not CooTensors; this does memory management
// However, when performing compute, just pass CooTensors, since they're lighter.
// The operator() is overloaded, so it's possible to also use/pass these as if they're CooTensors
struct CooTensorManager {
    std::shared_ptr<CooTensorUnique> tensor(new CooTensorUnique());

    /* utility functions */

    CooTensor operator() {
        return tensor->tensor;
    }


    /* parsing, conversion & creation functions */
    // TODO

};

#endif
