
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


struct COOTensor {
    CooPoint* points_h;
    CooPoint* points_d;
    PointSorting sorting;
    unsigned long long num_elements;
};

#endif
