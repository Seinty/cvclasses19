/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>
#include <random>   
#include <algorithm> 

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

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat gray = image.getMat();
    if (gray.channels() > 1)
        cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
    if (gray.type() != CV_8U)
        gray.convertTo(gray, CV_8U);

    const int desc_bytes   = 64;
    const int desc_bits    = desc_bytes * 8;
    const int pattern_radius = 15;

    static std::vector<cv::Point2i> pattern_a, pattern_b;
    if (pattern_a.empty())
    {
        std::mt19937 rng(0xFA57u);
        std::uniform_int_distribution<int> dist(-pattern_radius, pattern_radius);
        pattern_a.resize(desc_bits);
        pattern_b.resize(desc_bits);
        for (int i = 0; i < desc_bits; ++i)
        {
            pattern_a[i] = cv::Point2i(dist(rng), dist(rng));
            pattern_b[i] = cv::Point2i(dist(rng), dist(rng));
        }
    }

    descriptors.create(static_cast<int>(keypoints.size()), desc_bytes, CV_8U);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    auto sample = [&](int x, int y) -> int
    {
        x = std::min(std::max(x, 0), gray.cols - 1);
        y = std::min(std::max(y, 0), gray.rows - 1);
        return gray.at<uchar>(y, x);
    };

    for (std::size_t k = 0; k < keypoints.size(); ++k)
    {
        const int cx = static_cast<int>(keypoints[k].pt.x);
        const int cy = static_cast<int>(keypoints[k].pt.y);
        uchar* desc = desc_mat.ptr<uchar>(static_cast<int>(k));

        for (int i = 0; i < desc_bytes; ++i)
        {
            uchar byte = 0;
            for (int b = 0; b < 8; ++b)
            {
                const int idx = i * 8 + b;
                const int va  = sample(cx + pattern_a[idx].x, cy + pattern_a[idx].y);
                const int vb  = sample(cx + pattern_b[idx].x, cy + pattern_b[idx].y);
                if (va < vb)
                    byte |= static_cast<uchar>(1u << b);
            }
            desc[i] = byte;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors, bool /*= false*/)
{
    detect(image, keypoints, mask);
    compute(image, keypoints, descriptors);
}
} // namespace cvlib