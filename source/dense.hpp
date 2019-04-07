
#ifndef DENSE_HPP
#define DENSE_HPP

struct DenseTensor {
    float* values_h;
    float* values_d;
    unsigned int width, height, depth;
};



struct DenseMatrix {
    float* values_h;
    float* values_d;
    unsigned int height, width;
};

#endif
