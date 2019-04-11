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
//    http://tensor-compiler.org/codegen.html

    //https://sc17.supercomputing.org/SC17%20Archive/tech_poster/poster_files/post213s2-file2.pdf
    //tensor: I x J x K
    //matrix1: R x J
    //matrix2: R x K
    //result: R x I

    // Generated by the Tensor Algebra Compiler (tensor-compiler.org)
    for (int32_t pA = 0; pA < (A1_dimension * A2_dimension); pA++) {
        A_vals[pA] = 0.0;
    }
    for (int32_t pB1 = B1_pos[0]; pB1 < B1_pos[1]; pB1++) {
        int32_t iB = B1_coord[pB1];
        for (int32_t pB2 = B2_pos[pB1]; pB2 < B2_pos[(pB1 + 1)]; pB2++) {
            int32_t kB = B2_coord[pB2];
            for (int32_t pB3 = B3_pos[pB2]; pB3 < B3_pos[(pB2 + 1)]; pB3++) {
                int32_t lB = B3_coord[pB3];
                double tl = B_vals[pB3];
                for (int32_t jC = 0; jC < C2_dimension; jC++) {
                    int32_t pC2 = kB * C2_dimension + jC;
                    int32_t pD2 = lB * D2_dimension + jC;
                    int32_t pA2 = iB * A2_dimension + jC;
                    A_vals[pA2] = A_vals[pA2] + tl * C_vals[pC2] * D_vals[pD2];
                }
            }
        }
    }
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
    tensor->tensor.freeAllArrays();
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
    tensor->tensor.num_elements = nonZeroes;
    memcpy(tensor->tensor.points_h, matrixPoints.data(), sizeof(CooPoint) * matrixPoints.size());
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
