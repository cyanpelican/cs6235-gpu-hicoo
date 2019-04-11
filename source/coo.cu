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

void CooTensorManager::create(char *tensorFileName) {
    std::vector<CooPoint> matrixPoints;

    size_t nonZeroes = 0;
    std::string line;
    std::ifstream myfile(tensorFileName);

    //put all the points into a vector
    while (std::getline(myfile, line)) {
        ++nonZeroes;
        CooPoint currentPoint;
        std::vector<double> splitLine = split(&line, ' ');
        currentPoint.x = (unsigned int) splitLine[0];
        currentPoint.y = (unsigned int) splitLine[1];
        currentPoint.z = (unsigned int) splitLine[2];
        currentPoint.value = (float) splitLine[3];

        //This assumes there are 3 dimensions followed by one value
        matrixPoints.push_back(currentPoint);
    }

    matrixPoints.shrink_to_fit();

    //construct the COO object
    tensor->tensor.numElements = nonZeroes;
    tensor->tensor.points_h = matrixPoints.data();

}

std::vector<double> CooTensorManager::split(const std::string *str, char delimiter) {
    std::vector<double> internal;
    std::stringstream ss(*str); // Turn the string into a stream.
    std::string tok;

    while(getline(ss, tok, delimiter)) {
        internal.push_back(stod(tok));
    }

    return internal;
}
