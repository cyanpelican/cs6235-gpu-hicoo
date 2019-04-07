
#ifndef DENSE_HPP
#define DENSE_HPP

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
    void freeAllArrays() {
        free(points_h);
        cudaErrorCheck(cudaFree(tensor.points_d));
    }

    // safely uploads to gpu
    void uploadToDevice() {
        cudaErrorCheck(cudaFree(tensor.points_d));
        cudaErrorCheck(cudaMalloc((void **) &d_weight, sizeof(float)));
        cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(DensePoint) * num_elements, cudaMemcpyHostToDevice));
    }

    // safely downloads from gpu
    void downloadToHost() {
        free(points_h);
        points_h = malloc(sizeof(DensePoint) * num_elements);
        cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(DensePoint) * num_elements, cudaMemcpyDeviceToHost));
    }


    /* compute functions */
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
    void freeAllArrays() {
        free(points_h);
        cudaErrorCheck(cudaFree(tensor.points_d));
    }

    // safely uploads to gpu
    void uploadToDevice() {
        cudaErrorCheck(cudaFree(tensor.points_d));
        cudaErrorCheck(cudaMalloc((void **) &d_weight, sizeof(float)));
        cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(CooPoint) * num_elements, cudaMemcpyHostToDevice));
    }

    // safely downloads from gpu
    void downloadToHost() {
        free(points_h);
        points_h = malloc(sizeof(CooPoint) * num_elements);
        cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(CooPoint) * num_elements, cudaMemcpyDeviceToHost));
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

