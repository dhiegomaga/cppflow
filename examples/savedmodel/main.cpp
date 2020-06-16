#include "../../include/Model.h"
#include "../../include/Tensor.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <ctime> 
#include <iostream>

using namespace cv;

int main() {
    // Used variables
    const int IMG_SIZE = 150;
    std::vector<float> img_data(IMG_SIZE*IMG_SIZE*3);
    Mat image, preprocessed_image, flat;
    std::vector<float> predictions;
    std::string text;

    // Initialize neural network
    std::cout<<"Current tensorflow version: "<< TF_Version() << std::endl;
    Model m("model");

    // Input and output Tensors
    Tensor input(m, "serving_default_input_layer");
    Tensor prediction(m, "StatefulPartitionedCall");
    
    // Read image and convert to float
    image = cv::imread("intel-dataset/8.jpg");
    image.convertTo(image, CV_32F, 1.0/255.0);

    // Image dimensions    
    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();
    int total = image.total();

    // Assign to vector for 3 channel image
    // Souce: https://stackoverflow.com/a/56600115/2076973
    flat = image.reshape(1, image.total() * channels);
    img_data = image.isContinuous()? flat : flat.clone(); 

    // Feed data to input tensor
    input.set_data(img_data, {1, rows, cols, channels});
    
    // Run and show predictions
    m.run(input, prediction);
    
    // Get tensor with predictions
    predictions = prediction.Tensor::get_data<float>();
    for(int i=0; i<predictions.size(); i++) 
        std::cout<< std::to_string(predictions[i]) << std::endl;
    return 0;
}
