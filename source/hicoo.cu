#include "hicoo.hpp"
#include <assert.h>
#include "coo.hpp"


void HicooTensor::freeAllArrays() {
    DEBUG_PRINT("HT: free all arrays\n");
    freeHostArrays();
    freeDeviceArrays();
}
void HicooTensor::freeHostArrays() {
    DEBUG_PRINT("HT: free host arrays\n");
    DEBUG_PRINT("    - points_h = %p\n", points_h);
    DEBUG_PRINT("    - blocks_h = %p\n", blocks_h);
    free(points_h);
    free(blocks_h);
    points_h = nullptr;
    blocks_h = nullptr;
}
void HicooTensor::freeDeviceArrays() {
    DEBUG_PRINT("HT: free device arrays\n");
    DEBUG_PRINT("    - points_d = %p\n", points_d);
    DEBUG_PRINT("    - blocks_d = %p\n", blocks_d);
    if(points_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(points_d));
    if(blocks_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(blocks_d));
    points_d = nullptr;
    blocks_d = nullptr;
}

void HicooTensor::uploadToDevice() {
    DEBUG_PRINT("HT: upload to device\n");
    freeDeviceArrays();

    cudaErrorCheck(cudaMalloc((void **) &points_d, sizeof(HicooPoint) * numPoints));
    assert(points_d != nullptr);
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(HicooPoint) * numPoints, cudaMemcpyHostToDevice));

    cudaErrorCheck(cudaMalloc((void **) &blocks_d, sizeof(HicooBlock) * (numBlocks+1)));
    assert(blocks_d != nullptr);
    cudaErrorCheck(cudaMemcpy(blocks_d, blocks_h, sizeof(HicooBlock) * (numBlocks+1), cudaMemcpyHostToDevice));
}

void HicooTensor::downloadToHost() {
    DEBUG_PRINT("HT: download to host\n");
    freeHostArrays();

    points_h = (HicooPoint*)malloc(sizeof(HicooPoint) * numPoints);
    assert(points_h != nullptr);
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(HicooPoint) * numPoints, cudaMemcpyDeviceToHost));

    blocks_h = (HicooBlock*)malloc(sizeof(HicooBlock) * (numBlocks+1));
    assert(blocks_h != nullptr);
    cudaErrorCheck(cudaMemcpy(blocks_h, blocks_d, sizeof(HicooBlock) * (numBlocks+1), cudaMemcpyDeviceToHost));
}


CooTensorManager HicooTensor::toCoo() {
    DEBUG_PRINT("HT: to coo\n");
    CooTensorManager ret;
    assert(0); // TODO
    return ret;
}


DenseMatrixManager HicooTensor::mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    assert(points_h != nullptr);
    assert(blocks_h != nullptr);

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager HicooTensor::mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    assert(points_d != nullptr);
    assert(blocks_d != nullptr);

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager HicooTensor::mttkrp_fast(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}
