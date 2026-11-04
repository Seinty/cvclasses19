/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_corner_detector(int argc, char* argv[])
{

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
       std::cerr << "Cannot open camera" << std::endl;
       return -1;
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    //auto detector = cv::GFTTDetector::create(); // \todo use cvlib::corner_detector_fast
    auto detector = cvlib::corner_detector_fast::create();
    std::vector<cv::KeyPoint> corners;

    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        detector->detect(frame, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));
        utils::put_fps_text(frame, fps);
        // \todo add count of the detected corners at the top left corner of the image. Use green text color.
        cv::putText(frame, std::to_string(corners.size()), cv::Point(40,40), cv::FONT_ITALIC, 1.0, CV_RGB(0,255,0), 1);
        cv::imshow(demo_wnd, frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
