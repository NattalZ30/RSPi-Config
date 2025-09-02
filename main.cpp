#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Open default camera (0 = first webcam, 1 = second webcam, etc.)
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;  // Capture a new frame
        if (frame.empty()) {
            std::cerr << "Error: Empty frame grabbed." << std::endl;
            break;
        }

        cv::imshow("Webcam Feed", frame);

        // Press 'q' to exit
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
