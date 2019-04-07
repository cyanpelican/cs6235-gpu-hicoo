
#ifndef CSF_HPP
#define CSF_HPP

struct CsfPoint {
    float value;
    unsigned int fiberX, fiberY; // the other 2 coordinates not related to the fiber
};


struct CsfTensor {
    CsfPoint* points_h;
    CsfPoint* points_d;
    unsigned int* fiberAddresses_h; // should be length (numFibers+1); fiber ends are stored implicitly
    unsigned int* fiberAddresses_d;
    unsigned int numFibers;
    unsigned int width, height, depth;
    unsigned int mode;

    CsfTensor() {
        points_h = nullptr;
        points_d = nullptr;
        sorting = UNSORTED;
        num_elements = 0;
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


    /* compute functions */
    // TODO
};



// Don't make these. They're just middlemen.
struct CsfTensorUnique {
    CsfTensor tensor;

    CsfTensorUnique() {
        // nothing exciting to do
    }
    ~CsfTensorUnique() {
        tensor.freeAllArrays()
    }
};

// NOTE - build these, not CsfTensors; this does memory management
// However, when performing compute, just pass CsfTensors, since they're lighter.
// The operator() is overloaded, so it's possible to also use/pass these as if they're CsfTensors
struct CsfTensorManager {
    std::shared_ptr<CsfTensorUnique> tensor(new CsfTensorUnique());

    /* utility functions */

    CsfTensor operator() {
        return tensor->tensor;
    }


    /* parsing, conversion & creation functions */
    // TODO

};

// Note - to me, the "fiber" in csf makes me feel like x and y should be implicit and z should be explicit,
// rather than having x be implicit and y/z be explicit. But who knows.

#endif
