/**
 * @file preprocess.hpp
 * @author Aneesh Chodisetty (aneeshc@umd.edu)
 * @author Bhargav Kumar Soothram (bsoothra@umd.edu)
 * @author Joseph Pranadheer Reddy Katakam (jkatak@umd.edu)
 * @brief Header file for model.cpp
 * @version 0.1
 * @date 2022-10-10
 *
 * @copyright Copyright (c) 2022
 *
 */


#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

int main() {
    std::vector<std::string> class_names;
    std::ifstream ifs(std::string("coco.names").c_str());
    std::string line;
    while (getline(ifs, line)) {
        std::cout << line << std::endl;
        class_names.push_back(line);
    }
    // auto net = cv::dnn::readNetFromTensorflow("models/weights/frozen_inference_graph.pb",
                                // "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt");

    // load model weights and architecture
    std::string configuration = "models/yolov3.cfg";
    std::string model = "models/weights/yolov3.weights";

    // Load the network
    auto net = cv::dnn::readNetFromDarknet(configuration, model);
    cv::Mat frame, blob;

    // Set a min confidence score for the detections
    float min_confidence_score = 0.7;

    cv::VideoCapture cap(0);

    while (cap.isOpened()) {
        cv::Mat frame;
        bool isFrame = cap.read(frame);
        if (!isFrame) {
            break;
        }
        int frame_height = frame.cols;
        int frame_width = frame.rows;

        // cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5),
        //                         true, false);
        // net.setInput(blob);
        // cv::imshow("frame", frame);
        // cv::waitKey(1);
        auto start = cv::getTickCount();
        cv::dnn::blobFromImage(frame, blob, 1./255, cv::Size(frame_width, frame_height), \
                                cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);
        cv::Mat output = net.forward();
        // cv::imshow("Inferences", frame);
        // cv::waitKey(1);
        auto end = cv::getTickCount();

        // Matrix with all the detections
        cv::Mat results(output.size[2], output.size[3], CV_32F, output.ptr<float>());
        // Run through all the predictions
        for (int i = 0; i < results.rows; i++){
            int class_id = int(results.at<float>(i, 1));
            float confidence = results.at<float>(i, 2);
    
            // Check if the detection is over the min threshold and then draw bbox
            if (confidence > min_confidence_score){
                int bboxX = int(results.at<float>(i, 3) * frame.cols);
                int bboxY = int(results.at<float>(i, 4) * frame.rows);
                int bboxWidth = int(results.at<float>(i, 5) * frame.cols - bboxX);
                int bboxHeight = int(results.at<float>(i, 6) * frame.rows - bboxY);
                cv::rectangle(frame, cv::Point(bboxX, bboxY), cv::Point(bboxX + bboxWidth, bboxY + bboxHeight), cv::Scalar(0,0,255), 2);
                std::string class_name = class_names[class_id-1];
                cv::putText(frame, class_name + " " + std::to_string(int(confidence*100)) + "%", cv::Point(bboxX, bboxY - 10), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0,255,0), 2);
            }
        }
        

        auto totalTime = (end - start) / cv::getTickFrequency();
        

        cv::putText(frame, "FPS: " + std::to_string(int(1 / totalTime)), cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255, 0), 2, false);
        
        cv::imshow("image", frame);
        int k = cv::waitKey(1);
        if (k == 113){
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
  return 0;
}
