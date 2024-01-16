#include <iostream>
#include "trt_inference.h"

using namespace IMXAIEngine;
using namespace nvinfer1;
using namespace std;

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);
    TRT_Inference test1;
    std::vector<std::string> file_names;

    if(string( argv[1]) == "-s"){
        test1.trt_APIModel(argc, argv);
    }
    else if (string(argv[1]) == "-d")
    {
        test1.init_inference(string(argv[2]), argv[3],  file_names);
        cout <<"Dang thuc hien nha" << endl;
        test1.trt_detection(string( argv[3]), file_names);
    }
    

}