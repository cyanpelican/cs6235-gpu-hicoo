
#ifndef DENSE_HPP
#define DENSE_HPP

class DenseMatrix;
struct DenseTensor {
    float* values_h;
    float* values_d;
    unsigned int width, height, depth;

    DenseTensor() {
        points_h = nullptr;
        points_d = nullptr;
        sorting = UNSORTED;
        num_elements = 0;
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
    unsigned int height, width;

    DenseMatrix() {
        points_h = nullptr;
        points_d = nullptr;
        sorting = UNSORTED;
        num_elements = 0;
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
        tensor.freeAllArrays()
    }
};

// NOTE - build these, not DenseTensors; this does memory management
// However, when performing compute, just pass DenseTensors, since they're lighter.
// The operator() is overloaded, so it's possible to also use/pass these as if they're DenseTensors
struct DenseTensorManager {
    std::shared_ptr<DenseTensorUnique> tensor(new DenseTensorUnique());

    /* utility functions */

    DenseTensor operator() {
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
        tensor.freeAllArrays()
    }
};

// NOTE - build these, not DenseMatrixs; this does memory management
// However, when performing compute, just pass DenseMatrixs, since they're lighter.
// The operator() is overloaded, so it's possible to also use/pass these as if they're DenseMatrixs
struct DenseMatrixManager {
    std::shared_ptr<DenseMatrixUnique> tensor(new DenseMatrixUnique());

    /* utility functions */

    DenseMatrix operator() {
        return tensor->tensor;
    }


    /* parsing, conversion & creation functions */
    // TODO

};

#endif
