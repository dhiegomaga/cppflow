//
// Author : D Bersan
//

/*
 * Load model saved using tensorflow SavedModel format.
 *  
 * To access the names of the input and output tensors of the model, you can use the `saved_model_cli` tool, inside python bin/ folder. 
 * 
 * The syntax is (for the default tag-set and signature): `saved_model_cli show --dir /path/to/saved_model_folder/ --tag_set serve --signature_def serving_default`
 * 
 * > TODO Fix code to use other SignatureDefs other than the default one (It is also possible to define the name of the tensors via code while saving the model, but the documentation isn't so clear). 
 * 
 * More info at: https://stackoverflow.com/questions/58968918/accessing-input-and-output-tensors-of-a-tensorflow-2-0-savedmodel-via-the-c-api?noredirect=1#comment109422705_58968918
*/

#include "../../include/Model.h"
#include "../../include/Tensor.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <ctime> 
#include <iostream>

using namespace cv;

Mat preprocessImage(Mat image, int size);

int main() {
    // Used variables
    const int IMG_SIZE = 150;
    const std::string labels[] = {"sea", "glacier", "street", "forest", "buildings", "mountain"};
    std::vector<float> img_data(IMG_SIZE*IMG_SIZE*3);
    Mat image, preprocessed_image, flat;
    std::vector<float> predictions;
    std::string text;

    // Initialize neural network
    std::cout<<"Current tensorflow version: "<< TF_Version() << std::endl;
    Model m("../model");

    // Input and output Tensors
    Tensor input(m, "serving_default_input_layer");
    Tensor prediction(m, "StatefulPartitionedCall");
    
    for (int i=1; i<=6; i++) {
        // Read image
        image = cv::imread("../images/"+std::to_string(i)+".jpg");

        // Pre process        
        preprocessed_image = preprocessImage(image, IMG_SIZE);
        int rows = preprocessed_image.rows;
        int cols = preprocessed_image.cols;
        int channels = preprocessed_image.channels();

        // Assign to vector for 3 channel image
        // Souce: https://stackoverflow.com/a/56600115/2076973
        flat = preprocessed_image.reshape(1, preprocessed_image.total() * channels);
        img_data = preprocessed_image.isContinuous()? flat : flat.clone(); 

        // Feed data to input tensor
        input.set_data(img_data, {1, rows, cols, 3});
        
        // Run and show predictions
        m.run(input, prediction);
        
        // Get tensor with predictions
        predictions = prediction.Tensor::get_data<float>();
        int index = (int) std::distance(predictions.begin(),std::max_element(predictions.begin(), predictions.end()));
        auto probability = *std::max_element(predictions.begin(), predictions.end());
        text = labels[index];

        std::cout << "Predicted label: " << text << ", probability= " << std::to_string(probability) << std::endl;
    }

    return 0;
}

Mat preprocessImage(Mat image, int size){
    Mat result; 
    resize(image, result, cv::Size(size,size)); // resize to final size
    cvtColor(result, result, CV_BGR2RGB); // RGB
    result.convertTo(result, CV_32F, 1.0/255.0); // 0 - 1.0 range
    int c = result.channels();
    return result;
}