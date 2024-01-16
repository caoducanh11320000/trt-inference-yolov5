#include <iostream>
#include "trt_inference.h"



using namespace IMXAIEngine;
using namespace nvinfer1;

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);
    if(argv[1] == '-s'){
        // Goi HAm tạo .engine
    }
    else if (argv[1] == '-d')
    {
        // Goi ham Init và Do_inference
        // Luu y cac thong so truyen vao
    }
    

}