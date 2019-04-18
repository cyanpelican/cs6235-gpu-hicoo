#include "hicoo.hpp"
#include <assert.h>
#include "coo.hpp"

bool HicooBlock::operator<(const HicooBlock& b) {
    if(blockX < b.blockX) {
        return true;
    } else if(blockX > b.blockX) {
        return false;
    }
    if(blockY < b.blockY) {
        return true;
    } else if(blockY > b.blockY) {
        return false;
    }
    if(blockZ < b.blockZ) {
        return true;
    } else if(blockZ > b.blockZ) {
        return false;
    }

    return false;
}

void HicooTensor::freeAllArrays() {
    free(points_h);
    free(blocks_h);
    if(points_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(points_d));
    if(blocks_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(blocks_d));
    points_h = nullptr;
    blocks_h = nullptr;
    points_d = nullptr;
    blocks_d = nullptr;
}

void HicooTensor::uploadToDevice() {
    if(points_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaMalloc((void **) &points_d, sizeof(HicooPoint) * numPoints));
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(HicooPoint) * numPoints, cudaMemcpyHostToDevice));
    if(blocks_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(blocks_d));
    cudaErrorCheck(cudaMalloc((void **) &blocks_d, sizeof(HicooBlock) * (numBlocks+1)));
    cudaErrorCheck(cudaMemcpy(blocks_d, blocks_h, sizeof(HicooBlock) * (numBlocks+1), cudaMemcpyHostToDevice));
}

void HicooTensor::downloadToHost() {
    free(points_h);
    points_h = (HicooPoint*)malloc(sizeof(HicooPoint) * numPoints);
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(HicooPoint) * numPoints, cudaMemcpyDeviceToHost));
    free(blocks_h);
    blocks_h = (HicooBlock*)malloc(sizeof(HicooBlock) * (numBlocks+1));
    cudaErrorCheck(cudaMemcpy(blocks_h, blocks_d, sizeof(HicooBlock) * (numBlocks+1), cudaMemcpyDeviceToHost));
}


CooTensorManager HicooTensor::toCoo() {
    CooTensorManager ret;
    assert(0);
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
