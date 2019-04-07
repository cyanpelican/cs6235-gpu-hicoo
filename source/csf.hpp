
#ifndef CSF_HPP
#define CSF_HPP

struct CSFPoint {
    float value;
    unsigned int fiberX, fiberY; // the other 2 coordinates not related to the fiber
};


struct CSFTensor {
    CSFPoint* points;
    unsigned int* fiberAddresses; // should be length (numFibers+1); fiber ends are stored implicitly
    unsigned int numFibers;
    unsigned int width, height, depth;
    unsigned int mode;
};

// Note - to me, the "fiber" in csf makes me feel like x and y should be implicit and z should be explicit,
// rather than having x be implicit and y/z be explicit. But who knows.

#endif