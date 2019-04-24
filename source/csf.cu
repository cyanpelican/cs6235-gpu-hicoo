// Deprecated

#include "csf.hpp"
#include <assert.h>
#include "coo.hpp"

void CsfTensor::freeAllArrays() {
    free(points_h);
    free(fiberAddresses_h);
    if(points_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(points_d));
    if(fiberAddresses_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(fiberAddresses_d));
    points_h = nullptr;
    fiberAddresses_h = nullptr;
    points_d = nullptr;
    fiberAddresses_d = nullptr;
}

void CsfTensor::uploadToDevice() {
    if(points_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaMalloc((void **) &points_d, sizeof(CsfPoint) * numPoints()));
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(CsfPoint) * numPoints(), cudaMemcpyHostToDevice));
    if(fiberAddresses_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(fiberAddresses_d));
    cudaErrorCheck(cudaMalloc((void **) &fiberAddresses_d, sizeof(CsfPoint) * (numFibers()+1)));
    cudaErrorCheck(cudaMemcpy(fiberAddresses_d, fiberAddresses_h, sizeof(CsfPoint) * (numFibers()+1), cudaMemcpyHostToDevice));
}

void CsfTensor::downloadToHost() {
    free(points_h);
    points_h = (CsfPoint*)malloc(sizeof(CsfPoint) * numPoints());
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(CsfPoint) * numPoints(), cudaMemcpyDeviceToHost));
    free(fiberAddresses_h);
    fiberAddresses_h = (unsigned int*)malloc(sizeof(unsigned int) * (numFibers()+1));
    cudaErrorCheck(cudaMemcpy(fiberAddresses_h, fiberAddresses_d, sizeof(CsfPoint) * (numFibers()+1), cudaMemcpyDeviceToHost));
}


CooTensorManager CsfTensor::toCoo() {
    CooTensorManager ret;
    assert(0);
    return ret;
}


DenseMatrixManager CsfTensor::mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    assert(points_h != nullptr);
    assert(fiberAddresses_h != nullptr);

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager CsfTensor::mttkrp_naive_gpu(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;
    assert(points_d != nullptr);
    assert(fiberAddresses_d != nullptr);

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager CsfTensor::mttkrp_fast(DenseMatrix d, DenseMatrix c) {
    DenseMatrixManager ret;

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}
