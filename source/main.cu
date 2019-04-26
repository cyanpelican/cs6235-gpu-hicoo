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

using namespace std;

const int dimSize = 15;
int RANDOM_SEED = 1234;
bool cooKernal = 0, hicooKernal = 0;

vector <string> tensorList;


void compareOutput(DenseMatrix a, DenseMatrix b) {

    for (int i = 0; i < dimSize; i++) {
        for (int j = 0; j < dimSize; j++) {
            if (abs(a.access(i,j) - b.access(i,j)) > 1e-4) {
                printf("\n    Outputs do not match at index (%d,%d): %f vs %f", i,j, a.access(i,j), b.access(i,j));
                break;
            }
        }
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
	if (argc != 2) {
		fprintf(stderr, "Usage: <program name> <sparse matrix file name>\n");
		exit(1);
	}

	printf("========= Begin Test ========\n\n");

	printf("Adding tensor filenames to tensorList... ");
	tensorList.push_back("tensorA.file");
	tensorList.push_back("tensorB.file");
	tensorList.push_back("tensorC.file");
	printf("Done. (List Size = %d -> %s, %s, %s)\n",tensorList.size(),tensorList[0].c_str(),tensorList[1].c_str(),tensorList[2].c_str());


	//char *matrixName = argv[1];

	printf("Creating TensorManager Objects... ");
	CooTensorManager Coo;
	DenseTensorManager B;
	//HiooTensorManager Hicoo
	printf("Done.\n");



	printf("Creating CooTensor... ");
        Coo.tensor->tensor.setSize(dimSize*dimSize*dimSize,dimSize,dimSize,dimSize);
	printf("Done.\n");


	printf("Creating Random Dense Tensor (B) for testing... ");
	float testValue = 0, testI = 1, testJ = 2, testK = 3;  //Random point to sample
	B.tensor->tensor.setSize(dimSize,dimSize,dimSize);
	srand(RANDOM_SEED);
	for (int i = 0; i < dimSize; i++) {
	    for (int j = 0; j < dimSize; j++) {
		for (int k = 0; k < dimSize; k++) {
		     B.tensor->tensor.access(i,j,k) = rand() / (float) RAND_MAX;
		     if (i == testI && j == testJ && k == testK) { testValue = B.tensor->tensor.access(i,j,k); }
		}
	    }
	}
	printf("Done.\n");


	printf("Creating Random Dense Matrices (D,C) for testing... ");
	DenseMatrixManager D,C;
  DenseMatrix& c = C;
  DenseMatrix& d = D;
	d.setSize(dimSize,dimSize);
	c.setSize(dimSize,dimSize);
	for (int i = 0; i < dimSize; i++) {
	    for (int j = 0; j < dimSize; j++) {
		d.access(i,j) = rand() / (float) RAND_MAX;
		c.access(i,j) = rand() / (float) RAND_MAX;
	    }
	}
	printf("Done.\n");


	printf("Creating DenseMatrixManager for return Matrix (A)... ");
	DenseMatrixManager retDense;
	retDense.tensor->tensor.setSize(dimSize,dimSize);
	printf("Done.\n");
	printf("Creating CooMatrixManager for return Matrix (A)... ");
        DenseMatrixManager retCoo;
	printf("Done.\n");


	printf("\n=================== Beginning Kernel Tests on COO Tensor ===================\n\n");
	printf("Testing Dense Kernel Access function... ");
	float retValue = B.tensor->tensor.access(testI,testJ,testK);
	if (retValue == testValue) { printf("Passed. (%f and %f)\n",testValue,retValue); }
	else { printf("Failed. (Expected %f, returned %f)\n",testValue, retValue); }

	printf("Calculating MTTKRP (Dense)  value based on naive CPU Kernel (Ground Truth)... ");
	for (unsigned int i = 0; i < dimSize; i++) {
	for (unsigned int k = 0; k < dimSize; k++) {
        for (unsigned int l = 0; l < dimSize; l++) {
        for (unsigned int j = 0; j < d.height; j++) {
           retDense.tensor->tensor.access(i,j) = retDense.tensor->tensor.access(i,j) + B.tensor->tensor.access(i,k,l) * d.access(l,j) * c.access(k,j);
        }}}}
	printf("Done.\n");


	printf("Creating CooTensor from known data for comparison... ");
	srand(RANDOM_SEED);
	for (int i = 0; i < dimSize; i++) {
        for (int j = 0; j < dimSize; j++) {
        for (int k = 0; k < dimSize; k++) {
	   int idx = i*dimSize*dimSize + j*dimSize + k;
	   CooPoint p;
	   p.x = k; p.y = j; p.z = i;
	   p.value = rand() / (float) RAND_MAX;
           if(p.value > 1e-4) Coo.tensor->tensor.access(idx) = p;
        }}}
	printf("Done.\n");


	printf("Testing Dense to Coo conversion function... ");
	CooTensorManager CooComp = B.tensor->tensor.toCoo();
	bool mismatch = 0;
	for (int idx = 0; idx < dimSize*dimSize*dimSize; idx++) {
		CooPoint a, b;
		a = Coo.tensor->tensor.access(idx);
		b = CooComp.tensor->tensor.access(idx);
		if (a.x != b.x || a.y != b.y || a.z != b.z || a.value != b.value) {
			mismatch = 1;
			printf("    idx: %d  x: %d/%d  y: %d/%d  z: %d/%d  val: %d/%d\n",idx,a.x,b.x,a.y,b.y,a.z,b.z,a.value,b.value);
		}
	}
	if (mismatch) { printf("... Failed.\n"); }
	else { printf("Passed.\n"); }



	printf("Calculating MTTKRP (Coo) using implemented CPU kernel function call... ");
	retCoo = Coo.tensor->tensor.mttkrp_naive_cpu(D, C);
	printf("Done.\n");


	printf("Comparing Dense implementation to CPU Kernel Call (Ground truth vs Coo.naive_cpu)... ");
	compareOutput(retDense.tensor->tensor, retCoo.tensor->tensor);

  {
  	printf("Comparing Kevin's Dense implementation to CPU Kernel Call (Dense.naive_cpu vs Coo.naive_cpu)... ");
  	DenseMatrixManager retDenseK = B.tensor->tensor.mttkrp_naive_cpu(D,C);
  	compareOutput(retDenseK.tensor->tensor, retCoo.tensor->tensor);
  }

	printf("GROUND TRUTH ESTABLISHED\n");

	{
    printf("\nCalculating MTTKRP (Coo) using implemented GPU kernel function call... ");
  	DenseMatrixManager retCooGpu = Coo.tensor->tensor.mttkrp_naive_gpu(D,C); //COO GPU KERNEL

    printf("Comparing GPU Kernel Call to Ground Truth (Coo.naive_gpu vs Ground truth)... ");
  	compareOutput(retCoo.tensor->tensor, retCooGpu.tensor->tensor);
  }

  printf("Converting to hicoo\n");
  HicooTensorManager Hicoo = Coo.tensor->toHicoo();
  {
  	DenseMatrixManager retHicoo = Hicoo.tensor->tensor.mttkrp_naive_cpu(D, C);
  	printf("Testing Hicoo cpu MTTKRP... ");
    compareOutput(retCoo.tensor->tensor, retHicoo.tensor->tensor);
  }

  {
  	DenseMatrixManager retHicoo = Hicoo.tensor->tensor.mttkrp_naive_gpu(D, C);
  	printf("Testing Hicoo gpu MTTKRP... ");
    compareOutput(retCoo.tensor->tensor, retHicoo.tensor->tensor);
  }

	printf("\n=================== Beginning Kernel Tests on HiCOO Tensor ===================\n\n");


	/*printf("Creating HiCOO Tensor from known data for comparison... ");
        srand(RANDOM_SEED);
        for (int i = 0; i < dimSize; i++) {
        for (int j = 0; j < dimSize; j++) {
        for (int k = 0; k < dimSize; k++) {
           int idx = i*dimSize*dimSize + j*dimSize + k;
           CooPoint p;
           p.x = k; p.y = j; p.z = i;
           p.value = rand() / (float) RAND_MAX;
           if(p.value > 1e-4) Coo.tensor->tensor.access(idx) = p;
        }}}
        printf("Done.\n");


        printf("Testing Dense to Coo conversion function... ");
        CooTensorManager CooComp = B.tensor->tensor.toCoo();
        bool mismatch = 0;
        for (int idx = 0; idx < dimSize*dimSize*dimSize; idx++) {
                CooPoint a, b;
                a = Coo.tensor->tensor.access(idx);
                b = CooComp.tensor->tensor.access(idx);
                if (a.x != b.x || a.y != b.y || a.z != b.z || a.value != b.value) {
                        mismatch = 1;
                        printf("    idx: %d  x: %d/%d  y: %d/%d  z: %d/%d  val: %d/%d\n",idx,a.x,b.x,a.y,b.y,a.z,b.z,a.value,b.value);
                }
        }
        if (mismatch) { printf("... Failed.\n"); }
        else { printf("Passed.\n"); }



        printf("Calculating MTTKRP (Coo) using implemented CPU kernel function call... ");
        retCoo = Coo.tensor->tensor.mttkrp_naive_cpu(d, c);
        printf("Done.\n");


        printf("Comparing Dense implementation to CPU Kernel Call (Ground truth vs Coo.naive_cpu)... ");
        compareOutput(retDense.tensor->tensor, retCoo.tensor->tensor);

        printf("Comparing Kevin's Dense implementation to CPU Kernel Call (Dense.naive_cpu vs Coo.naive_cpu)... ");
        DenseMatrixManager retDenseK = B.tensor->tensor.mttkrp_naive_cpu(d,c);
        compareOutput(retDenseK.tensor->tensor, retCoo.tensor->tensor);

        printf("GROUND TRUTH ESTABLISHED\n");
        printf("\nCalculating MTTKRP (Coo) using implemented GPU kernel function call... ")
        retCoo = Coo.tensor->tensor.mttkrp_naive_gpu_wrapper(d,c);
        printf("Done\n");

        printf("Comparing GPU Kernel Call to Ground Truth (Coo.naive_gpu vs Ground truth)... ");
        compareOutput(retCoo.tensor->tensor, retDense.tensor->tensor);
	*/


	printf("That's a wrap\n");
	return 0;
}
