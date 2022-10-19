#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>


int main() {
    cv::VideoCapture cap(0);

    while (cap.isOpened()) {
        cv::Mat frame;
        bool isFrame = cap.read(frame);
        if (!isFrame) {
            break;
        }
        cv::imshow("frame", frame);
        cv::waitKey(1);
    }
}
