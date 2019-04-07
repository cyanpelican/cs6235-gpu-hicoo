
#ifndef DENSE_HPP
#define DENSE_HPP

struct DenseTensor {
    float* values;
    unsigned int width, height, depth;
};



struct DenseMatrix {
    float* values;
    unsigned int height, width;
};

#endif