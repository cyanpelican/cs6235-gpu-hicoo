
#ifndef COO_HPP
#define COO_HPP

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
        cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(CooPoint) * num_elements, cudaMemcpyDeviceToHost));
    }


    /* compute functions */
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
