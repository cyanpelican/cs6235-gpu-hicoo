#include "coo.hpp"
#include <assert.h>
#include "csf.hpp"
#include "hicoo.hpp"

void CooTensor::freeAllArrays() {
    free(points_h);
    cudaErrorCheck(cudaFree(points_d));
    points_h = nullptr;
    points_d = nullptr;
}

void CooTensor::uploadToDevice() {
    cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaMalloc((void **) &points_d, sizeof(CooPoint) * numElements));
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(CooPoint) * numElements, cudaMemcpyHostToDevice));
}

void CooTensor::downloadToHost() {
    free(points_h);
    points_h = (CooPoint*)malloc(sizeof(CooPoint) * numElements);
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(CooPoint) * numElements, cudaMemcpyDeviceToHost));
}


HicooTensorManager CooTensor::toHicoo() {
    HicooTensorManager ret;
    assert(0);
    return ret;
}
DenseTensorManager CooTensor::toDense() {
    DenseTensorManager ret;
    assert(0);
    return ret;
}
CsfTensorManager CooTensor::toCsf() {
    CsfTensorManager ret;
    assert(0);
    return ret;
}


DenseMatrixManager CooTensor::mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    assert(points_h != nullptr);

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager CooTensor::mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    assert(points_d != nullptr);

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager CooTensor::mttkrp_fast(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}
