#ifndef TRT_INFERENCE_H
#define TRT_INFERENCE_H

#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "model.h"

#include <iostream>
#include <chrono>
#include <cmath>

using namespace nvinfer1;

namespace IMXAIEngine
{

    typedef enum{
        TRT_RESULT_SUCCESS,
        TRT_RESULT_ERROR
    } trt_error;


    typedef struct
    {
        uint32_t ClassID;	
		float Confidence;	
        float bbox[4];
    } trt_results;
    
    typedef struct{
        std::vector<IMXAIEngine:: trt_results> results;
        int id;
    } trt_output;

    typedef struct
    {
        cv::Mat input_img;
        int id_img;
        
    } trt_input;
    

    class TRT_Inference
    {
    private:
        IRuntime* runtime= nullptr;
        ICudaEngine* engine= nullptr;
        IExecutionContext* context= nullptr;

    public:
        TRT_Inference();
        ~TRT_Inference(){
            if (context != nullptr)
            {
                context->destroy();
            }
            if (engine != nullptr)
            {
                engine->destroy();
            }
            if (runtime != nullptr)
            {
                runtime->destroy();
            }
            printf("Da huy Inference \n");
        }
        // trt_error init_inference(std::string engine_name ,const char * input_folder, std::vector<std::string> &file_names); 
        // trt_error trt_APIModel(std::string model_path);
        // trt_error trt_detection(std::vector<IMXAIEngine::trt_input> &trt_inputs, std::vector<IMXAIEngine::trt_output> &trt_outputs );

        trt_error init_inference(std::string engine_name ,const char * input_folder, std::vector<std::string> &file_names); 
        trt_error trt_APIModel  (int argc, char** argv);
        trt_error trt_detection (std::string img_dir , std::vector<std::string> &file_names);

    };

} // namespace IMXAIEngine

#endif