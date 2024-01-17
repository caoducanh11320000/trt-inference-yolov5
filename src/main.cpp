#include <iostream>
#include "trt_inference.h"

using namespace IMXAIEngine;
using namespace nvinfer1;
using namespace std;

std::vector<IMXAIEngine::trt_input> trt_inputs; 
std::vector<IMXAIEngine::trt_output> trt_outputs;

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
        //test1.trt_detection(string( argv[3]), file_names);

        //Phan moi them vao
        std::string folder= std::string(argv[3]); // luu ten thu muc chua anh
        for(int i=0; i< (int)file_names.size(); i++){

            std::cout << "Thuc hien voi anh:" << i <<std::endl;

            cv::Mat img = cv::imread(folder + "/" + file_names[i]);
            IMXAIEngine:: trt_input trt_input;
            if(!img.empty()) {
                //input_img.push_back(img); // danh so ID o day luon
                trt_input.input_img= img;
                trt_input.id_img = i;
                trt_inputs.push_back(trt_input);
                std::cout<< "thanh cong voi anh" << i <<std::endl;
                }
            else std::cout << "That bai" << std::endl;
        }

            test1.trt_detection(trt_inputs, trt_outputs);

    }
    

}