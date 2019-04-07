
#ifndef COO_HPP
#define COO_HPP

struct CooPoint {
    int x, y, z;
    float value;
};
enum PointSorting {
    UNSORTED,
    XYZ,
    Z_MORTON
};


struct COOTensor {
    Point* points;
    PointSorting sorting;
    int num_elements;
};

#endif
