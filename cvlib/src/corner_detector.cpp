/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
    cv::Mat gray = image.getMat();
    if (gray.channels() > 1)
        cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
    if (gray.type() != CV_8U)
        gray.convertTo(gray, CV_8U);

    const int threshold = 20;   // порог t
    const int N = 9;            // FAST-9

    static const cv::Point circle[16] = {
        { 0,-3}, { 1,-3}, { 2,-2}, { 3,-1},
        { 3, 0}, { 3, 1}, { 2, 2}, { 1, 3},
        { 0, 3}, {-1, 3}, {-2, 2}, {-3, 1},
        {-3, 0}, {-3,-1}, {-2,-2}, {-1,-3}
    };

    for (int y = 3; y < gray.rows - 3; ++y)
    {
        for (int x = 3; x < gray.cols - 3; ++x)
        {
            const int p    = gray.at<uchar>(y, x);
            const int high = p + threshold;
            const int low  = p - threshold;

            int brighter = 0, darker = 0;
            for (int i = 0; i < 16; i += 4)
            {
                const int v = gray.at<uchar>(y + circle[i].y, x + circle[i].x);
                brighter += (v > high);
                darker   += (v < low);
            }
            if (brighter < 3 && darker < 3)
                continue;

            bool is_corner = false;
            for (int start = 0; start < 16 && !is_corner; ++start)
            {
                int run_b = 0, run_d = 0;
                for (int k = 0; k < 16; ++k)
                {
                    const cv::Point& o = circle[(start + k) & 15];
                    const int v = gray.at<uchar>(y + o.y, x + o.x);
                    run_b = (v > high) ? run_b + 1 : 0;
                    run_d = (v < low)  ? run_d + 1 : 0;
                    if (run_b >= N || run_d >= N)
                    {
                        is_corner = true;
                        break;
                    }
                }
            }

            if (is_corner)
                keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y), 7.0f);
        }
    }
}

void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    std::srand(unsigned(std::time(0))); // \todo remove me
    // \todo implement any binary descriptor
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}
} // namespace cvlib