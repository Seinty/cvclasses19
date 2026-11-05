/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <string>

#include "utils.hpp"

int demo_feature_descriptor(int /*argc*/, char* /*argv*/[])
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
    auto detector_a = cvlib::corner_detector_fast::create();  // наш детектор
    auto detector_b = cv::ORB::create();                     // эталонный дескриптор
    std::vector<cv::KeyPoint> corners;

    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> frame;
        if (frame.empty())
            break;

        cv::imshow(main_wnd, frame);

        // \todo use your detector (detector_b)
        detector_a->detect(frame, corners);

        cv::Mat vis = frame.clone();
        cv::drawKeypoints(vis, corners, vis, cv::Scalar(0, 0, 255));

        // \todo add count of the detected corners at the top left corner of the image. Use green text color.
        cv::putText(vis,
                    "corners: " + std::to_string(static_cast<int>(corners.size())),
                    cv::Point(10, 70),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2);

        cv::imshow(demo_wnd, vis);

        pressed_key = cv::waitKey(30);

        // \todo draw histogram of SSD distribution for all descriptors instead of dumping into the file
        if (pressed_key == ' ')
        {
            if (corners.size() < 2)
            {
                std::cout << "Not enough keypoints for histogram\n";
                continue;
            }

            cv::Mat desc_a, desc_b;
            detector_a->compute(frame, corners, desc_a);   // наш,   32 байта (256 бит)
            detector_b->compute(frame, corners, desc_b);   // ORB,   32 байта (256 бит)

            auto draw_hamming_hist = [](const cv::Mat& desc,
                                        const std::string& title,
                                        cv::Scalar bar_color) -> cv::Mat
            {
                const int W = 640, H = 400, B = 64;
                cv::Mat img(H, W, CV_8UC3, cv::Scalar(40, 40, 40));

                if (desc.empty() || desc.rows < 2)
                {
                    cv::putText(img, title + ": <2 descriptors",
                                {10, 22}, cv::FONT_HERSHEY_SIMPLEX,
                                0.6, {255, 255, 255}, 1);
                    return img;
                }

                std::vector<float> d;
                d.reserve(desc.rows * (desc.rows - 1) / 2);
                for (int i = 0; i < desc.rows; ++i)
                {
                    for (int j = i + 1; j < desc.rows; ++j)
                    {
                        int hd = 0;
                        for (int k = 0; k < desc.cols; ++k)
                            hd += __builtin_popcount(
                                      static_cast<unsigned>(
                                          desc.at<uchar>(i, k) ^ desc.at<uchar>(j, k)));
                        d.push_back(static_cast<float>(hd));
                    }
                }

                const float range[]   = {0.f, 260.f};
                const float* ranges[] = {range};
                cv::Mat dmat(d);
                dmat = dmat.reshape(1, 1);

                cv::Mat hist;
                cv::calcHist(&dmat, 1, nullptr, cv::noArray(), hist, 1, &B, ranges);

                double max_h = 0.0;
                cv::minMaxLoc(hist, nullptr, &max_h);

                const int bin_w = W / B;
                if (max_h > 0.0)
                {
                    for (int i = 0; i < B; ++i)
                    {
                        const int h = static_cast<int>(
                                          hist.at<float>(i) / max_h * (H - 60));
                        cv::rectangle(img,
                                      {i * bin_w,     H - h},
                                      {(i + 1) * bin_w - 1, H - 1},
                                      bar_color, cv::FILLED);
                    }
                }

                cv::putText(img, title, {10, 22},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
                cv::putText(img,
                            "pairs: " + std::to_string(d.size()) +
                            "   max: " + std::to_string(static_cast<int>(max_h)) ,
                            {10, H - 10}, cv::FONT_HERSHEY_SIMPLEX,
                            0.5, {200, 200, 200}, 1);
                return img;
            };

            cv::Mat h_a = draw_hamming_hist(desc_a, "FAST descriptor (256 bit)", {0, 255, 0});
            cv::Mat h_b = draw_hamming_hist(desc_b, "ORB descriptor (256 bit)",  {0, 180, 255});

            cv::Mat both;
            cv::hconcat(h_a, h_b, both);
            cv::imshow("Hamming histograms", both);
        }

        std::cout << "Feature points: " << corners.size() << "\r" << std::flush;
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}