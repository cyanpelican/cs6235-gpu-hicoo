#include <assert.h>
#include <map>
#include "coo.hpp"
#include "csf.hpp"
#include "hicoo.hpp"


void CooTensor::freeAllArrays() {
    free(points_h);
    if(points_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(points_d));
    points_h = nullptr;
    points_d = nullptr;
}

void CooTensor::uploadToDevice() {
    if(points_d != nullptr) // Because the docs lie: "If devPtr is 0, no operation is performed."
        cudaErrorCheck(cudaFree(points_d));
    cudaErrorCheck(cudaMalloc((void **) &points_d, sizeof(CooPoint) * numElements));
    cudaErrorCheck(cudaMemcpy(points_d, points_h, sizeof(CooPoint) * numElements, cudaMemcpyHostToDevice));
}

// todo: is this ever necessary? The contents of the tensor are never changed. We should probably just free the device
//  memory
void CooTensor::downloadToHost() {
    free(points_h);
    points_h = (CooPoint*)malloc(sizeof(CooPoint) * numElements);
    cudaErrorCheck(cudaMemcpy(points_h, points_d, sizeof(CooPoint) * numElements, cudaMemcpyDeviceToHost));
}



// for std::map / std::set insertion
bool operator<(const HicooBlock& a, const HicooBlock& b) {
    if(a.blockX < b.blockX) {
        return true;
    } else if(a.blockX > b.blockX) {
        return false;
    }
    if(a.blockY < b.blockY) {
        return true;
    } else if(a.blockY > b.blockY) {
        return false;
    }
    if(a.blockZ < b.blockZ) {
        return true;
    } else if(a.blockZ > b.blockZ) {
        return false;
    }

    return false;
}
HicooTensorManager CooTensor::toHicoo(int blockWidth, int blockHeight, int blockDepth) {
    HicooTensorManager ret;
    HicooTensor retTensor = ret;

    // build an std::map of everything
    std::map<HicooBlock, std::vector<HicooPoint>> hicooMap;
    for(int i = 0; i < numElements; i++) {
        CooPoint p = access(i);

        HicooBlock block = {/*blockAddress =*/ 0,
            /*blockX =*/ (p.x)/blockWidth, /*blockX =*/ (p.y)/blockHeight, /*blockX =*/ (p.z)/blockDepth,
            /*UNUSED =*/ 0};

        HicooPoint hp = {/*x =*/ (unsigned char)((p.x)%blockWidth), /*y =*/ (unsigned char)((p.y)%blockHeight), /*z =*/ (unsigned char)((p.z)%blockDepth),
            /*UNUSED =*/ 0,
            /*value =*/ p.value};

        hicooMap[block].push_back(hp);
    }

    // TODO - put everything into
    retTensor.setSize(hicooMap.size(), numElements);

    unsigned int blockIndex = 0;
    unsigned long long pointIndex = 0;
    for(std::pair<HicooBlock, std::vector<HicooPoint>> pair : hicooMap) {
        retTensor.blocks_h[blockIndex] = pair.first;
        retTensor.blocks_h[blockIndex].blockAddress = pointIndex;
        for(HicooPoint p : pair.second) {
            retTensor.points_d[pointIndex++] = p;
        }
        blockIndex++;
    }

    // final element off the end of the list
    retTensor.blocks_h[blockIndex].blockAddress = pointIndex;
    retTensor.blocks_h[blockIndex].blockX = 0xFFFFFFFF;
    retTensor.blocks_h[blockIndex].blockY = 0xFFFFFFFF;
    retTensor.blocks_h[blockIndex].blockZ = 0xFFFFFFFF;
    retTensor.blocks_h[blockIndex].UNUSED = 0xFFFFFFFF;


    return ret;
}
DenseTensorManager CooTensor::toDense() {
    DenseTensorManager ret;
    ((DenseTensor)ret).setSize(width, height, depth);

    for(int i = 0; i < numElements; i++) {
        CooPoint p = access(i);
        ((DenseTensor)ret).access(p.x, p.y, p.z) = p.value;
    }

    return ret;
}
CsfTensorManager CooTensor::toCsf() {
    CsfTensorManager ret;
    assert(0);
    return ret;
}


DenseMatrixManager CooTensor::mttkrp_naive_cpu(DenseMatrix d, DenseMatrix c) {
    //order of dimensions goes height, width , depth
    //todo: double check this. It might be x,y,z: width, height, depth

    assert(this->points_h != nullptr);
    //check for compatible dimensions
    assert(this->width == d.width);
    assert(this->depth == c.width);

    DenseMatrixManager ret;
    ret.tensor->tensor.setSize(d.height, this->height);

    // http://tensor-compiler.org/codegen.html
    // https://sc17.supercomputing.org/SC17%20Archive/tech_poster/poster_files/post213s2-file2.pdf

    //the poster and the taco algorithm don't quite line up. This is the translated version:
    // tensor: i x k x l
    // matrix1: j x k
    // matrix2: j x l
    // result: j x i



    // Generated by the Tensor Algebra Compiler (tensor-compiler.org)
    // for (int32_t pA = 0; pA < (A1_dimension * A2_dimension); pA++) {
    //     A_vals[pA] = 0.0;
    // }
    // for (int32_t pB1 = B1_pos[0]; pB1 < B1_pos[1]; pB1++) {
    //     int32_t iB = B1_coord[pB1];
    //     for (int32_t pB2 = B2_pos[pB1]; pB2 < B2_pos[(pB1 + 1)]; pB2++) {
    //         int32_t kB = B2_coord[pB2];
    //         for (int32_t pB3 = B3_pos[pB2]; pB3 < B3_pos[(pB2 + 1)]; pB3++) {
    //             int32_t lB = B3_coord[pB3];
    //             double tl = B_vals[pB3];
    //             for (int32_t jC = 0; jC < C2_dimension; jC++) {
    //                 int32_t pC2 = kB * C2_dimension + jC;
    //                 int32_t pD2 = lB * D2_dimension + jC;
    //                 int32_t pA2 = iB * A2_dimension + jC;
    //                 A_vals[pA2] = A_vals[pA2] + tl * C_vals[pC2] * D_vals[pD2];
    //                 //my version:
    //                 *ret.tensor->tensor.access(i,j) = *ret.tensor->tensor.access(i,j) + this->access(i,k,l) * d.access(l,j) * c.access(k,j);
    //             }
    //         }
    //     }
    // }


   //A(i,j) = B(i,k,l) * C(k,j) * D(l,j)

    for (unsigned int i = 0; i < this->height; i++) {
        for (unsigned int k = 0; k < this->width; k++) {
            for (unsigned int l = 0; l < this->depth; l++) {
                for (unsigned int j = 0; j < d.height; j++) {
                    ret.tensor->tensor.access(i,j) = ret.tensor->tensor.access(i,j) + this->access(i,k,l) * d.access(l,j) * c.access(k,j);
                }
            }
        }
    }

    return ret;
}


//Not declared as part of the class... Cuda doesn't like it's kernels as part of OOP
__global__ void mttkrp_naive_gpu(CooTensor cooTensor, DenseMatrix d, DenseMatrix c, DenseMatrix ret) {
    assert(cooTensor.points_d != nullptr);
    //check for compatible dimensions
    assert(cooTensor.width == d.width);
    assert(cooTensor.depth == c.width);

    ret.setSize(d.height, cooTensor.height);

    if(blockDim.x * blockIdx.x + threadIdx.x < cooTensor.height * d.height) {
        //https://stackoverflow.com/a/29148148
        int index = blockDim.x * blockIdx.x + threadIdx.x;

        unsigned int i = index % cooTensor.height;
        unsigned int j = ((index - i) / cooTensor.height) % d.height;


        // We parallelize the 'i' and 'j' loops
        for (unsigned int k = 0; k < cooTensor.width; k++) {
            for (unsigned int l = 0; l < cooTensor.depth; l++) {
                //access will differentiate between host and device on its own
                ret.access(i, j) = ret.access(i, j) + cooTensor.access(i, k, l) * d.access(l, j) * c.access(k, j);
            }
        }

    }

    __syncthreads();
    
    // ret has already been written to. Now just return
}


//wrapper function for the sake of convenience
DenseMatrixManager CooTensor::mttkrp_naive_gpu_wrapper(DenseMatrix d, DenseMatrix c) {
    this->uploadToDevice();

    DenseMatrixManager ret;
    ret.tensor->tensor.uploadToDevice();
    d.uploadToDevice();
    c.uploadToDevice();

    mttkrp_naive_gpu<<<ceil(this->numElements/d.height), d.height>>>(&this, d, c, ret.tensor->tensor);
    cudaDeviceSynchronize();

    ret.tensor->tensor.downloadToHost();
    //todo: free the mem on the gpu?

//    this->downloadToHost(); //but like, why? The result is already in ret.
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
    int maxX = 0, maxY = 0, maxZ = 0;
    while (std::getline(myfile, line)) {
        ++nonZeroes;
        CooPoint currentPoint;
        std::vector<double> splitLine = split(&line, ' ');
        currentPoint.x = (unsigned int) splitLine[0];
        currentPoint.y = (unsigned int) splitLine[1];
        currentPoint.z = (unsigned int) splitLine[2];
        currentPoint.value = (float) splitLine[3];

        if(currentPoint.x > maxX) maxX = currentPoint.x;
        if(currentPoint.y > maxY) maxY = currentPoint.y;
        if(currentPoint.z > maxZ) maxZ = currentPoint.z;

        //This assumes there are 3 dimensions followed by one value
        matrixPoints.push_back(currentPoint);
    }

    matrixPoints.shrink_to_fit();

    //construct the COO object
    tensor->tensor.setSize(nonZeroes, maxX, maxY, maxZ);
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
