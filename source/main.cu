#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include "coo.hpp"
#include "common.hpp"
#include "dense.hpp"
#include "hicoo.hpp"

using namespace std;

const int dimSizeI = 30, dimSizeJ = 30, dimSizeK = 30, dimSizeL = 30;
int RANDOM_SEED = 1234;


template <typename classname, typename funcname>
float validateAndTime(classname inputTensor, DenseMatrixManager expected) {
    cudaEvent_t timing_start,timing_stop;

    cudaEventCreate(&timing_start);
    cudaEventCreate(&timing_stop);

    printf("  Calculating MTTKRP on class %s using %s... ", typeid(classname).name(), typeid(funcname).name());
    cudaEventRecord(timing_start,0);
    DenseMatrixManager result = inputTensor.tensor->tensor.funcname(D, C);
    cudaEventRecord(timing_stop);
    cudaEventSynchronize(timing_stop);
    cudaEventElapsedTime(&HicooCPUTime,timing_start, timing_stop);
    compareOutput(expected.tensor->tensor, result.tensor->tensor);
}

void compareOutput(DenseMatrix a, DenseMatrix b) {
    bool success = 1;
    for (int i = 0; i < dimSizeI; i++) {
        for (int j = 0; j < dimSizeJ; j++) {
            float mag = abs(a.access(i, j)) + 1e-4;
            if(abs(a.access(i, j) - b.access(i, j)) > mag * 1e-5) {
                printf("\n    Outputs do not match at index (%d,%d): %f vs %f", i,j, a.access(i,j), b.access(i,j));
                success = 0;
            }
        }
    }
    if (success) { printf("Passed.\n"); }
    else { printf("      FAILED :|\n"); }
}

void validateGroundTruth();
void testDenseToCoo(CooTensorManager Coo, int dimSizeI, int dimSizeK, int dimSizeL);

int main(int argc, char *argv[]) {

    int mode = 0;


    if (argc >= 2) {
        mode = 1;
    }

    printf("Creating TensorManager Objects... ");
    CooTensorManager Coo;
    DenseTensorManager B;
    HicooTensorManager Hicoo;
    printf("Done.\n");

    printf("Creating Timing Variables... ");
    cudaEvent_t timing_start,timing_stop;

    cudaEventCreate(&timing_start);
    cudaEventCreate(&timing_stop);
    printf("Done.\n");

    if (mode == 0) {

        printf("No command line arguments detected... Beginning generic testing sequence...\n\n");
        //exit(0);


        printf("Creating Random Dense Tensor (B) for testing... ");
        B.tensor->tensor.setSize(dimSizeI,dimSizeK,dimSizeL);
        srand(RANDOM_SEED);
        for (int i = 0; i < dimSizeI; i++) {
            for (int k = 0; k < dimSizeK; k++) {
                for (int l = 0; l < dimSizeL; l++) {
                     B.tensor->tensor.access(i,k,l) = rand() / (float) RAND_MAX;
                }
            }
        }
        printf("Done.\n");

        printf("Creating CooTensor... ");
        Coo.tensor->tensor.setSize(dimSizeI*dimSizeK*dimSizeL,dimSizeI,dimSizeK,dimSizeL);
        printf("Done.\n");
    }

    if (mode == 1) {
        //NEED TO CREATE TENSOR FROM FILEIN

        printf("Creating CooTensor from file '%s'... ", argv[1]);
        Coo.create(argv[1]);
        dimSizeI = Coo.tensor->tensor.depth;
        dimSizeK = Coo.tensor->tensor.height;
        dimSizeL = Coo.tensor->tensor.width;
        printf("Done.\n");
    }

    printf("=============================== Begin Test ================================\n\n");



    unsigned long long memUsage;


    printf("  Creating Random Dense Matrices (D,C) for testing... ");
    DenseMatrixManager D,C;
    DenseMatrix& c = C;
    DenseMatrix& d = D;
    d.setSize(dimSizeL,dimSizeJ);
    c.setSize(dimSizeK,dimSizeJ);
    for (int l = 0; l < dimSizeL; l++) {
        for (int j = 0; j < dimSizeJ; j++) {
            d.access(l,j) = rand() / (float) RAND_MAX;
        }
    }
    for (int k = 0; k < dimSizeK; k++) {
        for (int j = 0; j < dimSizeJ; j++) {
            c.access(k,j) = rand() / (float) RAND_MAX;
        }
    }
    printf("Done.\n");



    printf("\n=================== Beginning Kernel Tests on COO Tensor ===================\n\n");

    if (mode == 0) {
        testDenseToCoo(Coo, dimSizeI, dimSizeK, dimSizeL)
    }

    memUsage = Coo.tensor->tensor.getTotalMemory();
    printf("(Memory usage: %llu)\n",memUsage);

    printf("  Calculating MTTKRP (Coo) using implemented CPU kernel function call... ");
    //Time Sequential
    float CooCPUTime;
    cudaEventRecord(timing_start,0);
    DenseMatrixManager retCooCPU = Coo.tensor->tensor.mttkrp_naive_cpu(D, C);
    cudaEventRecord(timing_stop);
    cudaEventSynchronize(timing_stop);
    cudaEventElapsedTime(&CooCPUTime,timing_start, timing_stop);
    printf("Done.\n");

    printf("\n  Calculating MTTKRP (Coo) using implemented GPU kernel function call... ");
    //Time Parallel
    float CooGPUTime;
    cudaEventRecord(timing_start,0);
    DenseMatrixManager retCooGPU = Coo.tensor->tensor.mttkrp_naive_gpu(D, C); //COO GPU KERNEL
    cudaEventRecord(timing_stop);
    cudaEventSynchronize(timing_stop);
    cudaEventElapsedTime(&CooGPUTime,timing_start, timing_stop);
    printf("Done\n");


    if (mode == 0) {
        //DenseMatixManager Variables

        {
            printf("  Comparing Kevin's Dense implementation to CPU Kernel Call (Dense.naive_cpu vs Coo.naive_cpu)... ");
            DenseMatrixManager retDense = B.tensor->tensor.mttkrp_naive_cpu(D,C);
            compareOutput(retDense.tensor->tensor, retCooCPU.tensor->tensor);
        }
    }

    // TODO - this
    float f = validateAndTime<CooTensorManager, mttkrp_naive_gpu>(Coo, retCooCPU);

    {
        printf("\n  Calculating MTTKRP (Coo) using implemented GPU kernel function call... ");
        DenseMatrixManager retCooGpu = Coo.tensor->tensor.mttkrp_naive_gpu(D,C); //COO GPU KERNEL

        printf("  Comparing GPU Kernel Call to Ground Truth (Coo.naive_gpu vs Ground truth)... ");
        compareOutput(retCooCPU.tensor->tensor, retCooGpu.tensor->tensor);
    }


    printf("\n=================== Beginning Kernel Tests on HiCOO Tensor ===================\n\n");

    printf("  Converting to hicoo\n");
    Hicoo = Coo.tensor->tensor.toHicoo();
    float HicooCPUTime;
    {
        //Time Sequential
        printf("  Calculating MTTKRP (HiCOO) using implemented CPU kernel function call... ");
        cudaEventRecord(timing_start,0);
        DenseMatrixManager retHicoo = Hicoo.tensor->tensor.mttkrp_naive_cpu(D, C);
        cudaEventRecord(timing_stop);
        cudaEventSynchronize(timing_stop);
        cudaEventElapsedTime(&HicooCPUTime,timing_start, timing_stop);
        compareOutput(retCooCPU.tensor->tensor, retHicoo.tensor->tensor);
    }

    float HicooGPUTime;
    {
        //Time Parallel
        printf("  Comparing Hicoo implementation to CPU Kernel Call (Ground truth vs HiCoo.naive_cpu)... ");
        cudaEventRecord(timing_start,0);
        DenseMatrixManager retHicoo = Hicoo.tensor->tensor.mttkrp_naive_gpu(D, C);
        cudaEventRecord(timing_stop);
        cudaEventSynchronize(timing_stop);
        cudaEventElapsedTime(&HicooGPUTime,timing_start,timing_stop);
        compareOutput(retCooCPU.tensor->tensor, retHicoo.tensor->tensor);
    }

    printf("\n  ==================== Memory Usage ======================= \n");

    if (mode == 0) {
        memUsage = B.tensor->tensor.getTotalMemory();
        printf("  Dense Tensor (%d,%d,%d) --> %llu B\n",dimSizeI,dimSizeK,dimSizeL,memUsage);
    }

    memUsage = Coo.tensor->tensor.getTotalMemory();
    printf("  COO Tensor (%d,%d,%d) --> %llu B\n",dimSizeI,dimSizeK,dimSizeL,memUsage);

    memUsage = Hicoo.tensor->tensor.getTotalMemory();
    printf("  HiCOO Tensor (%d,%d,%d) --> %llu B\n",dimSizeI,dimSizeK,dimSizeL,memUsage);

    printf("  =========================================================\n\n");

    printf("\n  ======================= Timing(s) ======================= \n");

    printf("  COO MTTKRP (%d,%d,%d)\n",dimSizeI,dimSizeK,dimSizeL);
    printf("    CPU -> %f\n", CooCPUTime);
    printf("    GPU -> %f\n", CooGPUTime);
    printf("    Speedup -> %f\n\n", CooCPUTime/CooGPUTime);
    printf("  HiCOO MTTKRP (%d,%d,%d)\n",dimSizeI,dimSizeK,dimSizeL);
    printf("    CPU -> %f\n", HicooCPUTime);
    printf("    GPU -> %f\n", HicooGPUTime);
    printf("    Speedup -> %f\n", HicooCPUTime/HicooGPUTime);

    printf("  =========================================================\n\n");
    printf("That's a wrap\n");
    return 0;
}

void testDenseToCoo(CooTensorManager Coo, int dimSizeI, int dimSizeK, int dimSizeL) {
    printf("  Creating CooTensor from known data for comparison... ");
    srand(RANDOM_SEED);
    for (int i = 0; i < dimSizeI; i++) {
        for (int k = 0; k < dimSizeK; k++) {
            for (int l = 0; l < dimSizeL; l++) {
                int idx = i*dimSizeK*dimSizeL + k*dimSizeL + l;
                CooPoint p;
                p.x = l; p.y = k; p.z = i;
                p.value = rand() / (float) RAND_MAX;
                if(p.value > 1e-4) Coo.tensor->tensor.access(idx) = p;
            }
        }
    }
    printf("Done. ");


    printf("  Testing Dense to Coo conversion function... ");
    CooTensorManager CooComp = B.tensor->tensor.toCoo();
    bool mismatch = 0;
    for (int idx = 0; idx < dimSizeI*dimSizeK*dimSizeL; idx++) {
        CooPoint a, b;
        a = Coo.tensor->tensor.access(idx);
        b = CooComp.tensor->tensor.access(idx);
        if (a.x != b.x || a.y != b.y || a.z != b.z || a.value != b.value) {
            mismatch = 1;
            //printf("    idx: %d  x: %d/%d  y: %d/%d  z: %d/%d  val: %d/%d\n",idx,a.x,b.x,a.y,b.y,a.z,b.z,a.value,b.value);
        }
    }
    if (mismatch) { printf("... Failed.\n"); }
    else { printf("Passed.\n"); }

}

void validateGroundTruth() {

    /*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-   MATLAB TENSOR / MATRIX VALIDATION CODE =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/

    // STRUCTURE OF MATLAB TENSOR:

    // y = i, x = j, z = k;
    //    /    j
    //   ============//
    //   ============//
    // i ============// k
    //   ============//
    //   ============/



    DenseTensorManager matlab;
    matlab.tensor->tensor.setSize(3,3,3);

    matlab.tensor->tensor.access(0,0,0) = 0.8311;
    matlab.tensor->tensor.access(0,0,1) = 0.3952;
    matlab.tensor->tensor.access(0,0,2) = 0.4412;

    matlab.tensor->tensor.access(0,1,0) = 0.5568;
    matlab.tensor->tensor.access(0,1,1) = 0.2911;
    matlab.tensor->tensor.access(0,1,2) = 0.2135;

    matlab.tensor->tensor.access(0,2,0) = 0.2345;
    matlab.tensor->tensor.access(0,2,1) = 0.2098;
    matlab.tensor->tensor.access(0,2,2) = 0.1484;

    matlab.tensor->tensor.access(1,0,0) = 0.2844;
    matlab.tensor->tensor.access(1,0,1) = 0.2804;
    matlab.tensor->tensor.access(1,0,2) = 0.0949;

    matlab.tensor->tensor.access(1,1,0) = 0.3379;
    matlab.tensor->tensor.access(1,1,1) = 0.9659;
    matlab.tensor->tensor.access(1,1,2) = 0.7877;

    matlab.tensor->tensor.access(1,2,0) = 0.4038;
    matlab.tensor->tensor.access(1,2,1) = 0.0240;
    matlab.tensor->tensor.access(1,2,2) = 0.6363;

    matlab.tensor->tensor.access(2,0,0) = 0.3720;
    matlab.tensor->tensor.access(2,0,1) = 0.6422;
    matlab.tensor->tensor.access(2,0,2) = 0.0034;

    matlab.tensor->tensor.access(2,1,0) = 0.9030;
    matlab.tensor->tensor.access(2,1,1) = 0.4056;
    matlab.tensor->tensor.access(2,1,2) = 0.8192;

    matlab.tensor->tensor.access(2,2,0) = 0.3261;
    matlab.tensor->tensor.access(2,2,1) = 0.7646;
    matlab.tensor->tensor.access(2,2,2) = 0.5833;


    DenseMatrixManager mD, mC;
    mD.tensor->tensor.setSize(3,3);
    mC.tensor->tensor.setSize(3,3);

    mD.tensor->tensor.access(0,0) = 0.2061;
    mD.tensor->tensor.access(0,1) = 0.8238;
    mD.tensor->tensor.access(0,2) = 0.0042;
    mD.tensor->tensor.access(1,0) = 0.7055;
    mD.tensor->tensor.access(1,1) = 0.7682;
    mD.tensor->tensor.access(1,2) = 0.4294;
    mD.tensor->tensor.access(2,0) = 0.9975;
    mD.tensor->tensor.access(2,1) = 0.3894;
    mD.tensor->tensor.access(2,2) = 0.3276;

    mC.tensor->tensor.access(0,0) = 0.7853;
    mC.tensor->tensor.access(0,1) = 0.9508;
    mC.tensor->tensor.access(0,2) = 0.3240;
    mC.tensor->tensor.access(1,0) = 0.4353;
    mC.tensor->tensor.access(1,1) = 0.7073;
    mC.tensor->tensor.access(1,2) = 0.7889;
    mC.tensor->tensor.access(2,0) = 0.7104;
    mC.tensor->tensor.access(2,1) = 0.1381;
    mC.tensor->tensor.access(2,2) = 0.2877;

    DenseMatrixManager matlabComp = matlab.tensor->tensor.mttkrp_naive_cpu(mD,mC);
    printf("Output of MTTKRP on Dense Matrix from MATLAB values:\n");

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%f ", matlabComp.tensor->tensor.access(i,j));
        }
        printf("\n");
    }

    /* =========   OUTPUT FROM ABOVE CALC    =========

    1.175773 1.701301 0.298766
    1.466742 1.484061 0.644793
    1.824243 1.883446 0.592149

    /* =========   MTTKRP CODE FROM MATLAB:  =========

    n = 1
    KRP = khatrirao(D,C); %<--Khatri-Rao product, omitting U{2}
    M = permute(X.data, [n:size(X,n), 1:n-1]);
    M = reshape(M,size(X,n),[]); %<--Matricized tensor data
    M*KRP

    ans =

    1.1757    1.7013    0.2988
    1.4666    1.4841    0.6449
    1.8243    1.8836    0.5922


    exit(0);

    =================================================*/
}
