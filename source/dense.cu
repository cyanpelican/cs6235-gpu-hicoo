#include "dense.hpp"
#include "coo.hpp"

void DenseTensor::freeAllArrays() {
    DEBUG_PRINT("DT: freeing all arrays\n");
    freeHostArrays();
    freeDeviceArrays();
}
void DenseTensor::freeHostArrays() {
    DEBUG_PRINT("DT: freeing host arrays\n");
    DEBUG_PRINT("    - values_h = %p\n", values_h);
    free(values_h);
    values_h = nullptr;
}
void DenseTensor::freeDeviceArrays() {
    DEBUG_PRINT("DT: freeing device arrays\n");
    DEBUG_PRINT("    - values_d = %p\n", values_d);
    if(values_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(values_d));
    values_d = nullptr;
}

// safely uploads to gpu
void DenseTensor::uploadToDevice() {
    DEBUG_PRINT("DT: upload to device\n");
    freeDeviceArrays();
    cudaErrorCheck(cudaMalloc((void **) &values_d, sizeof(float) * width*height*depth));
    assert(values_d != nullptr);
    cudaErrorCheck(cudaMemcpy(values_d, values_h, sizeof(float) * width*height*depth, cudaMemcpyHostToDevice));
}

// safely downloads from gpu
void DenseTensor::downloadToHost() {
    DEBUG_PRINT("DT: download to host\n");
    freeHostArrays();
    values_h = (float*)malloc(sizeof(float) * width*height*depth);
    assert(values_h != nullptr);
    cudaErrorCheck(cudaMemcpy(values_h, values_d, sizeof(float) * width*height*depth, cudaMemcpyDeviceToHost));
}


void DenseMatrix::freeAllArrays() {
    DEBUG_PRINT("DM: freeing all arrays\n");
    freeHostArrays();
    freeDeviceArrays();
}
void DenseMatrix::freeHostArrays() {
    DEBUG_PRINT("DM: freeing host arrays\n");
    DEBUG_PRINT("    - values_h = %p\n", values_h);
    free(values_h);
    values_h = nullptr;
}
void DenseMatrix::freeDeviceArrays() {
    DEBUG_PRINT("DM: freeing device arrays\n");
    DEBUG_PRINT("    - values_d = %p\n", values_d);
    if(values_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(values_d));
    values_d = nullptr;
}

// safely uploads to gpu
void DenseMatrix::uploadToDevice() {
    DEBUG_PRINT("DM: upload to device\n");
    assert(values_h != nullptr);
    freeDeviceArrays();
    cudaErrorCheck(cudaMalloc((void **) &values_d, sizeof(float) * width*height));
    assert(values_d != nullptr);
    cudaErrorCheck(cudaMemcpy(values_d, values_h, sizeof(float) * width*height, cudaMemcpyHostToDevice));
}

// safely downloads from gpu
void DenseMatrix::downloadToHost() {
    DEBUG_PRINT("DM: download to host\n");
    assert(values_d != nullptr);
    freeHostArrays();
    values_h = (float*)malloc(sizeof(float) * width*height);
    assert(values_h != nullptr);
    cudaErrorCheck(cudaMemcpy(values_h, values_d, sizeof(float) * width*height, cudaMemcpyDeviceToHost));
}


CooTensorManager DenseTensor::toCoo(float epsilon) {
    DEBUG_PRINT("DT: to coo (epsilon = %f)\n", epsilon);
    CooTensorManager ret;
    CooTensor& tensor = ret;

    DEBUG_PRINT("    - count nnz\n");
    unsigned long long numNonzeros = 0;
    for(int i = 0; i < depth; i++) {
        for(int j = 0; j < height; j++) {
            for(int k = 0; k < width; k++) {
                if(abs(access(i, j, k)) > epsilon) {
                    numNonzeros++;
                }
            }
        }
    }

    DEBUG_PRINT("    - realloc\n");
    tensor.setSize(numNonzeros, depth, height, width);
    tensor.sorting = XYZ;

    // convert
    DEBUG_PRINT("    - final conversion\n");
    unsigned long long ptIdx = 0;
    for(int i = 0; i < depth; i++) {
        for(int j = 0; j < height; j++) {
            for(int k = 0; k < width; k++) {
                if(abs(access(i, j, k)) > epsilon) {
                    tensor.access(ptIdx).value = access(i, j, k);
                    tensor.access(ptIdx).x = k;
                    tensor.access(ptIdx).y = j;
                    tensor.access(ptIdx).z = i;
                    ptIdx++;
                }
            }
        }
    }

    return ret;
}


DenseMatrixManager DenseTensor::mttkrp_naive_cpu(DenseMatrixManager d, DenseMatrixManager c) {
    DEBUG_PRINT("DT: mttkrp naive cpu\n");
    DEBUG_PRINT("    - asserts, initialization\n");
    DenseMatrixManager ret;
    DenseMatrix& a = ret;
    assert(values_h != nullptr);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);
    int I = this->depth, J = d.width, K = this->height, L = this->width;
    DEBUG_PRINT("    - I = %d, J = %d, K = %d, L = %d\n", I, J, K, L);
    assert(d.height == L);
    assert(c.height == K);
    assert(c.width  == J);

    DEBUG_PRINT("    - malloc output\n");
    a.setSize(I, J);

    DEBUG_PRINT("    - compute\n");
    for(int i = 0; i < I; i++) {
        for(int j = 0; j < J; j++) {
            for(int k = 0; k < K; k++) {
              for(int l = 0; l < L; l++) {
                  a.access(i, j) += access(i,k,l) * d.access(l,j) * c.access(k,j);
              }
            }
        }
    }


    return ret;
}

DenseMatrixManager DenseTensor::mttkrp_naive_gpu(DenseMatrixManager d, DenseMatrixManager c) {
    DEBUG_PRINT("DT: mttkrp naive gpu\n");
    DenseMatrixManager ret;
    assert(values_d != nullptr);

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}

DenseMatrixManager DenseTensor::mttkrp_fast(DenseMatrixManager d, DenseMatrixManager c) {
    DEBUG_PRINT("DT: mttkrp fast\n");
    DenseMatrixManager ret;

    // TODO
    assert(0);

    // A(i,j) = B(i,k,l) * D(l,j) * C(k,j);

    return ret;
}


void DenseMatrix::setSize_d(unsigned int height, unsigned int width) {
    DEBUG_PRINT("DM: setSize_d (h %d, w %d)\n", height, width);
    freeDeviceArrays();
    cudaErrorCheck(cudaMalloc((void **) &values_d, sizeof(float) * width*height));
    assert(values_d != nullptr);
    cudaErrorCheck(cudaMemset(values_d, 0.0f, width*height * sizeof(float)));
    this->width = width;
    this->height = height;
}
