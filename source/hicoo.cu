#include "hicoo.hpp"
#include <assert.h>

void HicooTensor::freeAllArrays() {
    free(points_h);
    free(blocks_h);
    cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaFree(blocks_d));
    points_h = nullptr;
    blocks_h = nullptr;
    points_d = nullptr;
    blocks_d = nullptr;
}

void HicooTensor::uploadToDevice() {
    cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaMalloc((void **) &points_d, sizeof(HicooPoint) * numPoints));
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(HicooPoint) * numPoints, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaFree(blocks_d));
    cudaErrorCheck(cudaMalloc((void **) &blocks_d, sizeof(HicooBlock) * numBlocks));
    cudaErrorCheck(cudaMemcpy(blocks_d, blocks_h, sizeof(HicooBlock) * numBlocks, cudaMemcpyHostToDevice));
}

void HicooTensor::downloadToHost() {
    free(points_h);
    points_h = (HicooPoint*)malloc(sizeof(HicooPoint) * numPoints);
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(HicooPoint) * numPoints, cudaMemcpyDeviceToHost));
    free(blocks_h);
    blocks_h = (HicooBlock*)malloc(sizeof(HicooBlock) * numBlocks);
    cudaErrorCheck(cudaMemcpy(blocks_h, blocks_d, sizeof(HicooBlock) * numBlocks, cudaMemcpyDeviceToHost));
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
