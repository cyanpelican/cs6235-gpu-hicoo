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

int dimSizeI = 30, dimSizeJ = 30, dimSizeK = 30, dimSizeL = 30;
float density = 1.0f;
int RANDOM_SEED = 1234;

void compareOutput(DenseMatrix a, DenseMatrix b) {
    int errors = 0;
    const int maxErrors = 50;
    long aZeros = 0, bZeros = 0;

    DEBUG_PRINT("Performing validation...\n");
    DEBUG_PRINT("Sample data: a(0,0)=%f, b(0,0)=%f\n", a.access(0,0), b.access(0,0));

    assert(a.values_h != nullptr);
    assert(b.values_h != nullptr);
    for (int i = 0; i < dimSizeI; i++) {
        for (int j = 0; j < dimSizeJ; j++) {
            float mag = abs(a.access(i, j)) + 1e-4;
            if(abs(a.access(i, j) - b.access(i, j)) > mag * 1e-5) {
                printf("\n    Outputs do not match at index (%d,%d): %f vs %f", i,j, a.access(i,j), b.access(i,j));
                errors++;

                if(errors > maxErrors) {
                    printf("      FAILED, and stopped printing after %d errors.\n", maxErrors);
                    fflush(stdout);
                    return;
                }
            }

            if(abs(a.access(i, j)) < 1e-4) {
                aZeros += 1;
            }
            if(abs(b.access(i, j)) < 1e-4) {
                bZeros += 1;
            }
        }
    }
    if (errors==0) { printf("Passed.\n"); }
    else { printf("\n      FAILED :|\n"); }

    if(aZeros > (a.width * a.height) * .25) {
      printf("There seem to be a lot of zeros in the A matrix.\n");
    }
    if(bZeros > (b.width * b.height) * .25) {
      printf("There seem to be a lot of zeros in the B matrix.\n");
    }
    fflush(stdout);

    DEBUG_PRINT("done with compareOutput\n");
}

void validateGroundTruth();
void performAndTestDenseToCoo(CooTensorManager Coo, DenseTensorManager B);
template <typename Class, typename Functype>
float validateAndTime(Class inputTensor, Functype func, std::string funcname, DenseMatrixManager D, DenseMatrixManager C, DenseMatrixManager expected);

#define FUNC_AND_NAME(func) &func, #func

int main(int argc, char *argv[]) {
    // args: [J] [BS] [TensorFilepath] [NOCPU?]
    bool useDense = false;
    bool allowCPU = true;
    int  blockSize = 4;
    float FOREVER = 9e9;

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

    if (argc >= 2) {
        // read J
        dimSizeJ = atoi(argv[1]);
    }
    if (argc >= 3) {
        // read BS
        blockSize = atoi(argv[2]);
    }

    if (argc >= 4 && std::string(argv[3]).rfind("dense-", 0) != 0) {
        //NEED TO CREATE TENSOR FROM FILEIN

        printf("Creating CooTensor from file '%s'... ", argv[3]);
        fflush(stdout);
        Coo.create(argv[3]);
        dimSizeI = Coo.tensor->tensor.depth;
        dimSizeK = Coo.tensor->tensor.height;
        dimSizeL = Coo.tensor->tensor.width;
        printf("Done.\n");
        fflush(stdout);
    } else {
        // Generate dense tensor
        useDense = true;

        if(argc >= 4) { // if we're passed a dense-WIDTHxHEIGHTxDEPTHdDENSITY string in place of a filename
            std::string denseSizeStr = std::string(argv[3]).substr(6); // crop off 'dense-'

            std::string dim1Str; // https://stackoverflow.com/questions/236129/how-do-i-iterate-over-the-words-of-a-string
            std::string dim2Str;
            std::string dim3Str;
            std::string densityStr;
            std::stringstream ss(denseSizeStr);
            std::getline(ss, dim1Str, 'x');
            std::getline(ss, dim2Str, 'x');
            std::getline(ss, dim3Str, 'd');
            std::getline(ss, densityStr, 'd');
            dimSizeL = atoi(dim1Str.c_str());
            dimSizeK = atoi(dim2Str.c_str());
            dimSizeI = atoi(dim3Str.c_str());
            if(densityStr.length() > 0)
                density  = atof(densityStr.c_str());

            printf("Creating dense tensor of size %d x %d x %d, density = %f\n\n", dimSizeL, dimSizeK, dimSizeI, density);
        } else {
            printf("No filename detected... Beginning generic testing sequence...\n\n");
        }
        //exit(0);


        printf("Creating Random Dense Tensor (B) for testing... ");
        B.tensor->tensor.setSize(dimSizeI,dimSizeK,dimSizeL);
        srand(RANDOM_SEED);
        for (int i = 0; i < dimSizeI; i++) {
            for (int k = 0; k < dimSizeK; k++) {
                for (int l = 0; l < dimSizeL; l++) {
                    if (density >= 1.0f || (rand() / (float) RAND_MAX) <= density)
                        B.tensor->tensor.access(i,k,l) = rand() / (float) RAND_MAX;
                }
            }
        }
        printf("Done.\n");

        printf("Creating CooTensor... ");
        fflush(stdout);
        Coo.tensor->tensor.setSize(dimSizeI*dimSizeK*dimSizeL,dimSizeI,dimSizeK,dimSizeL);
        printf("Done.\n");
        fflush(stdout);

        if(density != 1.0f) {
            printf("Skipping dense-to-coo validation because density != 1\n");
            Coo = B.tensor->tensor.toCoo();
        } else {
            performAndTestDenseToCoo(Coo, B);
        }
    }


    if (argc >= 5) {
        if(strcmp(argv[4], "NOCPU") == 0) {
            allowCPU = false;
        }
    }

    printf("=============================== Begin Test ================================\n\n");



    unsigned long long memUsage;


    printf("  Creating Random Dense Matrices (D,C) for testing... ");
    fflush(stdout);
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
    printf("Uploading to device... ");
    d.uploadToDevice();
    c.uploadToDevice();
    printf("Done.\n");
    fflush(stdout);


    DenseMatrixManager goodRet;
    printf("\n=================== Beginning Kernel Tests on COO Tensor ===================\n\n");
    Coo.tensor->tensor.uploadToDevice();

    memUsage = Coo.tensor->tensor.getTotalMemory();
    printf("(Memory usage: %llu)\n",memUsage);
    fflush(stdout);


    // Time COO to use as comparison
    float CooCPUTime = FOREVER;
    if(allowCPU) {
        printf("  Calculating MTTKRP (Coo) using implemented CPU kernel function call... ");
        fflush(stdout);
        cudaEventRecord(timing_start,0);
        goodRet = Coo.tensor->tensor.mttkrp_naive_cpu(D, C);
        cudaEventRecord(timing_stop);
        cudaEventSynchronize(timing_stop);
        cudaEventElapsedTime(&CooCPUTime,timing_start, timing_stop);
        printf("    Time = %f\n", CooCPUTime);
        printf("Done.\n");
        fflush(stdout);
    } else {
        printf("WARNING - VALIDATING AGAINST A GPU RUN, BECAUSE CPU IS TOO SLOW\n");
        fflush(stdout);

        goodRet = Coo.tensor->tensor.mttkrp_naive_gpu(D, C);
        goodRet.tensor->tensor.downloadToHost();
        goodRet.tensor->tensor.freeDeviceArrays();
    }

    // Time Parallel
    float CooGPUTime = validateAndTime(Coo, FUNC_AND_NAME(CooTensor::mttkrp_naive_gpu), D, C, goodRet);

    float CooKevin1Time = validateAndTime(Coo, FUNC_AND_NAME(CooTensor::mttkrp_kevin1), D, C, goodRet);

    Coo.tensor->tensor.freeDeviceArrays();

    if (useDense && allowCPU) {
        printf("\n=================== Beginning Kernel Tests on Dense Tensor ===================\n\n");
        fflush(stdout);

        float denseCpuTime = validateAndTime(B, FUNC_AND_NAME(DenseTensor::mttkrp_naive_cpu), D, C, goodRet);

        //float denseGpuTime = validateAndTime(B, FUNC_AND_NAME(DenseTensor::mttkrp_naive_gpu), D, C, goodRet);

        fflush(stdout);
    }



    printf("\n=================== Beginning Kernel Tests on HiCOO Tensor ===================\n\n");

    printf("  Converting to hicoo\n");
    fflush(stdout);
    Hicoo = Coo.tensor->tensor.toHicoo(blockSize, blockSize, blockSize);
    Hicoo.tensor->tensor.uploadToDevice();

    float HicooCPUTime = FOREVER;
    if(allowCPU) {
        HicooCPUTime = validateAndTime(Hicoo, FUNC_AND_NAME(HicooTensor::mttkrp_naive_cpu), D, C, goodRet);
    }

    float HicooGPUTime = validateAndTime(Hicoo, FUNC_AND_NAME(HicooTensor::mttkrp_naive_gpu), D, C, goodRet);

    // TODO - PUT OPTIMIZED KERNELS HERE

    float HicooKevin1Time = validateAndTime(Hicoo, FUNC_AND_NAME(HicooTensor::mttkrp_kevin1), D, C, goodRet);

    float HicooKevin2Time = validateAndTime(Hicoo, FUNC_AND_NAME(HicooTensor::mttkrp_kevin2), D, C, goodRet);

    float HicooKevin3Time = validateAndTime(Hicoo, FUNC_AND_NAME(HicooTensor::mttkrp_kevin3), D, C, goodRet);

    Hicoo.tensor->tensor.freeDeviceArrays();



    printf("\n  ==================== Memory Usage ======================= \n");

    if (useDense) {
        memUsage = B.tensor->tensor.getTotalMemory();
        printf("  Dense Tensor (%d,%d,%d) --> %llu B\n",dimSizeI,dimSizeK,dimSizeL,memUsage);
    }

    memUsage = Coo.tensor->tensor.getTotalMemory();
    printf("  COO Tensor (%d,%d,%d) --> %llu B\n",dimSizeI,dimSizeK,dimSizeL,memUsage);

    memUsage = Hicoo.tensor->tensor.getTotalMemory();
    printf("  HiCOO Tensor (%d,%d,%d) --> %llu B\n",dimSizeI,dimSizeK,dimSizeL,memUsage);

    printf("  =========================================================\n\n");

    printf("\n  ======================= Timing(s) ======================= \n");

    printf("  COO MTTKRP (%d,%d,%d; J=%d)\n",dimSizeI,dimSizeK,dimSizeL, dimSizeJ);
    printf("    CPU -> %f\n", CooCPUTime);
    printf("    NAIVE GPU -> %f\n", CooGPUTime);
    printf("      Speedup -> %f\n", CooCPUTime/CooGPUTime);
    printf("    Kevin1 -> %f\n", CooKevin1Time);
    printf("      Speedup -> %f\n", CooCPUTime/CooKevin1Time);
    printf("\n");
    printf("  HiCOO MTTKRP (%d,%d,%d J=%d)\n",dimSizeI,dimSizeK,dimSizeL, dimSizeJ);
    printf("    CPU -> %f\n", HicooCPUTime);
    printf("    NAIVE GPU -> %f\n", HicooGPUTime);
    printf("      Speedup -> %f\n", HicooCPUTime/HicooGPUTime);
    // TODO - PRINT TIME FOR OPTIMIZED KERNELS HERE
    printf("    Kevin1 -> %f\n", HicooKevin1Time);
    printf("      Speedup -> %f\n", CooCPUTime/HicooKevin1Time);
    printf("    Kevin2 -> %f\n", HicooKevin2Time);
    printf("      Speedup -> %f\n", CooCPUTime/HicooKevin2Time);
    printf("    Kevin3 -> %f\n", HicooKevin3Time);
    printf("      Speedup -> %f\n", CooCPUTime/HicooKevin3Time);
    printf("\n");

    printf("  =========================================================\n\n");
    printf("That's a wrap\n\n\n\n");
    fflush(stdout);
    return 0;
}







// helper functions

void performAndTestDenseToCoo(CooTensorManager Coo, DenseTensorManager B) {
    assert(density == 1.0f); // not implemented for other densities
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
    printf("Done.\n");


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

#include <cxxabi.h>
#include <execinfo.h>
template <typename T>
std::string demangledClassName(T o) {
    // https://gcc.gnu.org/onlinedocs/libstdc++/manual/ext_demangling.html

    int status;
    char* demangled = abi::__cxa_demangle(typeid(o).name(), 0, 0, &status);
    std::string ret = demangled;
    free(demangled);

    return ret;
}


template <typename Class, typename Functype>
float validateAndTime(Class inputTensor, Functype func, std::string funcname, DenseMatrixManager D, DenseMatrixManager C, DenseMatrixManager expected) {
    // minor black magic from https://timmurphy.org/2014/08/28/passing-member-functions-as-template-parameters-in-c/
    DEBUG_PRINT("Running validateAndTime...\n");
    float retTime;
    cudaEvent_t timing_start,timing_stop;

    cudaEventCreate(&timing_start);
    cudaEventCreate(&timing_stop);

    std::string classname = demangledClassName(inputTensor);
    printf("  Calculating MTTKRP on class %s using %s... ", classname.c_str(), funcname.c_str());

    // compute
    DEBUG_PRINT("Launching compute...\n");
    cudaEventRecord(timing_start,0);
    DenseMatrixManager result = (inputTensor.tensor->tensor.*func)(D, C);
    cudaEventRecord(timing_stop);
    cudaEventSynchronize(timing_stop);

    cudaEventElapsedTime(&retTime, timing_start, timing_stop);

    printf("    Time = %f\n", retTime);
    fflush(stdout);

    DEBUG_PRINT("Download result...\n");
    if(result.tensor->tensor.values_h == nullptr) {
        result.tensor->tensor.downloadToHost();
    }

    DEBUG_PRINT("Validating...\n");
    compareOutput(expected.tensor->tensor, result.tensor->tensor);

    DEBUG_PRINT("done with validateAndTime\n");
    return retTime;
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
