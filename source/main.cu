#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include "coo.hpp"
#include "common.hpp"



int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <program name> <sparse matrix file name>\n");
        //todo: I could extend this to take in the 2 dense matrices file names as well
        exit(1);
    }

    // a note on mttkrp dimension requirements:
    // tensor: i x k x l
    // matrix1: j x k
    // matrix2: j x l
    // result: j x i

    char *matrixName = argv[1];
    CooTensorManager coo;
    coo.create(argv[1]);

    DenseMatrix c;
    //c.create(argv[2]);
    DenseMatrix d;
    //d.create(argv[3]);


    DenseMatrix cpu_coo_result = coo.tensor->tensor.mttkrp_naive_cpu(d, c);

    coo.tensor->tensor.uploadToDevice();
    DenseMatrix gpu_coo_result = coo.tensor->tensor.mttkrp_naive_gpu_wrapper(d, c);

//    std::cout << ((CooTensor)Coo).numElements << std::endl;


    return 0;
}
