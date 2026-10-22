/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <cmath>
#include <algorithm>
#include <map>
#include <vector>

namespace
{
struct descriptor : public std::vector<double>
{
    using std::vector<double>::vector;
    descriptor operator-(const descriptor& right) const
    {
        descriptor temp = *this;
        for (size_t i = 0; i < temp.size(); ++i)
        {
            temp[i] -= right[i];
        }
        return temp;
    }

    double norm_l1() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += std::abs(v);
        }
        return res;
    }

    double norm_l2() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += v * v;
        }
        return res;
    }
};

int round_to_odd(double d)
{
    int r = static_cast<int>(std::round(d));
    if (r % 2 == 0)
        ++r;
    return std::max(r, 1);
}

// Верхняя граница размера ядра.
const int MAX_KERNEL_SIZE = 21;

const double THETAS[]  = {0.0, CV_PI / 4, CV_PI / 2, 3 * CV_PI / 4};
const double LAMBDAS[] = {4.0, 7.0, 10.0};
const double GAMMAS[]  = {0.5};
const double SIGMAS[]  = {2.5, 5.0};

const int N_THETAS  = sizeof(THETAS)  / sizeof(THETAS[0]);
const int N_LAMBDAS = sizeof(LAMBDAS) / sizeof(LAMBDAS[0]);
const int N_GAMMAS  = sizeof(GAMMAS)  / sizeof(GAMMAS[0]);
const int N_SIGMAS  = sizeof(SIGMAS)  / sizeof(SIGMAS[0]);

std::vector<cv::Mat> makeKernels(int kernel_size)
{
    std::vector<cv::Mat> kernels;
    kernels.reserve(N_THETAS * N_LAMBDAS * N_GAMMAS * N_SIGMAS);

    const double max_sigma = kernel_size / 4.0;

    for (int t = 0; t < N_THETAS; ++t)
    {
        for (int l = 0; l < N_LAMBDAS; ++l)
        {
            for (int g = 0; g < N_GAMMAS; ++g)
            {
                for (int s = 0; s < N_SIGMAS; ++s)
                {
                    if (SIGMAS[s] > max_sigma)
                        continue;

                    kernels.push_back(
                        cv::getGaborKernel(
                            cv::Size(kernel_size, kernel_size),
                            SIGMAS[s], THETAS[t], LAMBDAS[l], GAMMAS[g]));
                }
            }
        }
    }

    if (kernels.empty())
    {
        kernels.push_back(
            cv::getGaborKernel(
                cv::Size(kernel_size, kernel_size),
                max_sigma, THETAS[0], LAMBDAS[0], GAMMAS[0]));
    }

    return kernels;
}

const std::vector<cv::Mat>& getKernels(int kernel_size)
{
    static std::map<int, std::vector<cv::Mat>> cache;

    auto it = cache.find(kernel_size);
    if (it != cache.end())
        return it->second;

    return cache.emplace(kernel_size, makeKernels(kernel_size)).first->second;
}

void calculateDescriptor(const cv::Mat& image, int kernel_size, descriptor& descr)
{
    const auto& kernels = getKernels(kernel_size);

    descr.clear();
    descr.reserve(2 * kernels.size());

    cv::Mat response;      
    cv::Scalar mean;       
    cv::Scalar dev;        

    for (const auto& kernel : kernels)
    {
        cv::filter2D(image, response, CV_32F, kernel);
        cv::meanStdDev(response, mean, dev);
        descr.push_back(mean[0]);
        descr.push_back(dev[0]);
    }
}
} // namespace

namespace cvlib
{
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi_in, double eps)
{
    cv::Mat res = cv::Mat::zeros(image.size(), CV_8UC1);

    cv::Rect roi = roi_in & cv::Rect(0, 0, image.cols, image.rows);
    if (roi.width < 4 || roi.height < 4)
        return res;

    cv::Mat image_f;
    image.convertTo(image_f, CV_32F);

    int kernel_size = round_to_odd(std::min(roi.height, roi.width) / 2.0);
    kernel_size = std::min(kernel_size, MAX_KERNEL_SIZE);

    descriptor reference;
    calculateDescriptor(image_f(roi), kernel_size, reference);

    descriptor test(reference.size());
    cv::Rect baseROI = roi - roi.tl();

    const int step = 7;
    const double eps2 = eps * eps;

    for (int i = 0; i <= image.cols - roi.width; i += step)
    {
        for (int j = 0; j <= image.rows - roi.height; j += step)
        {
            cv::Rect curROI = baseROI + cv::Point(i, j);
            calculateDescriptor(image_f(curROI), kernel_size, test);

            if ((test - reference).norm_l2() <= eps2)
                res(curROI) = 255;
        }
    }

    return res;
}
} // namespace cvlib