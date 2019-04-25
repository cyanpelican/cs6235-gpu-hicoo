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

const int dimSize = 10;
int RANDOM_SEED = 1234;
bool cooKernal = 0, hicooKernal = 0;

vector <string> tensorList;


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
	

    // 	a note on mttkrp dimension requirements:
    // 	tensor: i x k x l
    // 	matrix1: j x k
    // 	matrix2: j x l
    // 	result: j x i

    //	char *matrixName = argv[1];
    //	CooTensorManager coo;
    //	coo.create(argv[1]);

    //	DenseMatrix c;
    //	c.create(argv[2]);
    //	DenseMatrix d;
    //	d.create(argv[3]);


        //DenseMatrix cpu_coo_result = coo.tensor->tensor.mttkrp_naive_cpu(d, c);

        //coo.tensor->tensor.uploadToDevice();
        //DenseMatrix gpu_coo_result = coo.tensor->tensor.mttkrp_naive_gpu_wrapper(d, c);

        //std::cout << ((CooTensor)Coo).numElements << std::endl;
      
	printf("Creating CooTensor... ");
        Coo.tensor->tensor.setSize(dimSize*dimSize*dimSize,dimSize,dimSize,dimSize);
	
	printf("Done.\n");

	printf("Creating Random Dense Tensor (B) for testing... ");
	//Random Dense Tensor
	
	float testValue = 0, testI = 1, testJ = 2, testK = 3;
	//float B[dimSize][dimSize][dimSize];
	B.tensor->tensor.setSize(dimSize,dimSize,dimSize);
	srand(RANDOM_SEED);
	for (int i = 0; i < dimSize; i++) {
		for (int j = 0; j < dimSize; j++) {
			for (int k = 0; k < dimSize; k++) {
				B.tensor->tensor.access(i,j,k) = rand() / 1000000;
				if (i == testI && j == testJ && k == testK) { testValue = B.tensor->tensor.access(i,j,k); }
				//Sets a random Coo Tensor instead
				/*int idx = i*dimSize*dimSize + j*dimSize + k;
				pointList[idx].x = i;
				pointList[idx].y = j;
				pointList[idx].z = k;
				pointList[idx].value = rand() / 100000;*/
			}
		}
	}
	printf("Done.\n");


	//Random Dense Matrices
	printf("Creating Random Dense Matrices (D,C) for testing... ");
	//float d[dimSize][dimSize];
	//float c[dimSize][dimSize];
	DenseMatrix d,c;
	d.setSize(dimSize,dimSize);
	c.setSize(dimSize,dimSize);
	for (int i = 0; i < dimSize; i++) {
		for (int j = 0; j < dimSize; j++) {
			d.access(i,j) = rand() / 1000000;
			c.access(i,j) = rand() / 1000000;
		}
	}
	//DenseMatrixManager D,C;
	//D.tensor->tensor.setSize(dimSize,dimSize);
	//C.tensor->tensor.setSize(dimSize,dimSize);
 
	printf("Done.\n");

	printf("Creating DenseMatrixManager for return Matrix (A)... ");
	DenseMatrixManager retDense;
	retDense.tensor->tensor.setSize(dimSize,dimSize);
	printf("Done.\n");
	printf("Creating CooMatrixManager for return Matrix (A)... ");
        //DenseMatrixManager retCoo;
        //retCoo.tensor->tensor.setSize(dimSize,dimSize);
	printf("Done.\n");

	printf("\n========= Beginning Kernel Tests on Tensor =========\n\n");
	//Dense Test
	//Calculate MTTKRP on CPU
	//float cpuOut[dimSize][dimSize];
	printf("Testing Dense Kernel Access function... ");
	float retValue = B.tensor->tensor.access(testI,testJ,testK);
	if (retValue == testValue) { printf("Passed. (%f and %f)\n",testValue,retValue); }
	else { printf("Failed. (Expected %f, returned %f)\n",testValue, retValue); }
	
	printf("Calculating MTTKRP (Dense)  value based on naive CPU Kernel... ");
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
	   p.value = rand() / 100000;
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
			//printf("    idx: %d  x: %d/%d  y: %d/%d  z: %d/%d  val: %d/%d\n",idx,a.x,b.x,a.y,b.y,a.z,b.z,a.value,b.value);
		}
	}
	if (mismatch) { printf("... Failed.\n"); }
	else { printf("Passed.\n"); }

	printf("Calculating MTTKRP (Coo) using implemented kernel function call... ");	
	DenseMatrixManager retCoo = Coo.tensor->tensor.mttkrp_naive_cpu(d, c);
	printf("Done.\n");

	printf("Comparing Kernel Call to Dense implementation... ");
        //VERIFY CORRECTNESS BY COMPARING OUTPUTS
     	for (int i = 0; i < dimSize; i++) {
        for (int j = 0; j < dimSize; j++) {
	  if (abs(retDense.tensor->tensor.access(i,j) - retCoo.tensor->tensor.access(i,j)) > 1e-4) {
          printf("\n    Outputs do not match at index (%d,%d): %f vs %f", i,j, retDense.tensor->tensor.access(i,j), retCoo.tensor->tensor.access(i,j));
	  break;// exit(2);
          }
        }}
	printf("\n");
	retDense = B.tensor->tensor.mttkrp_naive_cpu(d,c);

	printf("Comparing Kernel Call to Kevin's Dense implementation... ");
        //VERIFY CORRECTNESS BY COMPARING OUTPUTS
        for (int i = 0; i < dimSize; i++) {
        for (int j = 0; j < dimSize; j++) {
          if (abs(retDense.tensor->tensor.access(i,j) - retCoo.tensor->tensor.access(i,j)) > 1e-4) {
          printf("\n    Outputs do not match at index (%d,%d): %f vs %f", i,j, retDense.tensor->tensor.access(i,j), retCoo.tensor->tensor.access(i,j));
          break;//exit(2);
          }
        }}
	printf("\n");
	//Calculate MTTKRP on GPU
	//float gpuOut[dimSize][dimSize];


	//Coo Test
//	Dm = 
//	if (cooKernal) {
//		//Dense to Coo, repeat
//		Dense = ((CooTensor)Coo).mttkrp_naive_cpu(D, C);
//		//((CooTensor)Coo).points_h = &pointList;
//	}
//
	//Hicoo Test
//	if (hicooKernal) {
//		//Coo to Hicoo, repeat
//		//Dense = 
//	}
//
	//Coo.create(argv[1]);
//	std::cout << ((CooTensor)Coo).numElements << std::endl;


	printf("That's a wrap\n");
	return 0;
}