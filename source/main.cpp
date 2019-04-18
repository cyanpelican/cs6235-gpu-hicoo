#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include "coo.hpp"
#include "common.hpp"

int size = 50;
int RANDOM_SEED = 1234;
bool cooKernal = 0, hicooKernal = 0;

vector <string> tensorList;

tensorList.push_back("tensorA.file");
tensorList.push_back("tensorB.file");
tensorList.push_back("tensorC.file");

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <program name> <sparse matrix file name>\n");
        exit(1);
    }

    char *matrixName = argv[1];
    CooTensorManager Coo;
    DenseTensorManager Dense;
    HiooTensorManager Hicoo;
    
    
    CooPoint pointList[size*size*size];

    //Random Dense Tensor
    float B[size][size][size];
    srand(RANDOM_SEED);
    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            for (int k=0; k<size; k++) {
                B[i][j][k] = rand() / 100000;
                //Sets a random Coo Tensor instead
                /*int idx = i*size*size + j*size + k;
                pointList[idx].x = i;
                pointList[idx].y = j;
                pointList[idx].z = k;
                pointList[idx].value = rand() / 100000;*/
            }
        }
    }

    //Random Dense Matrices
    float D[size][size];
    float C[size][size];
    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            D[i][j] = rand() / 100000;
            C[i][j] = rand() / 100000;
        }
    }

    //Dense Test
    //Calculate MTTKRP on CPU
    float cpuOut[size][size];
    //Calculate MTTKRP on GPU
    float gpuOut[size][size];

    // VERIFY CORRECTNESS BY COMPARING OUTPUTS
    for (i=0; i<size; i++) { // minibatch size
        for (j=0; j<size; j++) { // output feature map
            if(abs(cpuOut[i][j]-gpuOut[i][j]) > .0001) {
                printf("Outputs do not match! - %f vs %f\n", abs(cpuOut[i][j]), abs(gpuOut[i][j]));
                exit(2);
            }
        }
    }

    //Coo Test
    if (cooKernal) {
        //Dense = ((CooTensor)Coo).mttkrp_
        //((CooTensor)Coo).points_h = &pointList;
    }
    
    //Hicoo Test
    if (hicooKernal) {
        Dense = 
    }

    //Coo.create(argv[1]);
    std::cout << ((CooTensor)Coo).numElements << std::endl;

    
    
    return 0;
}

