/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
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
    auto detector_a = cvlib::corner_detector_fast::create();
    auto detector_b = cv::KAZE::create(false, true); // upright=true
    
    std::vector<cv::KeyPoint> corners;
    std::vector<cv::KeyPoint> kaze_corners;

    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> frame;
        if (frame.empty())
            break;

        cv::imshow(main_wnd, frame);

        detector_a->detect(frame, corners);
        detector_b->detect(frame, kaze_corners);

        cv::Mat vis = frame.clone();
        cv::drawKeypoints(vis, corners, vis, cv::Scalar(0, 0, 255));

        cv::putText(vis,
                    "FAST: " + std::to_string(static_cast<int>(corners.size())) +
                    "  KAZE: " + std::to_string(static_cast<int>(kaze_corners.size())),
                    cv::Point(10, 70),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2);

        cv::imshow(demo_wnd, vis);

        pressed_key = cv::waitKey(30);

        if (pressed_key == ' ')
        {
            if (corners.size() < 2 || kaze_corners.size() < 2)
            {
                std::cout << "Not enough keypoints for histogram\n";
                continue;
            }

            // --- FAST-дескрипторы (ВАШ КАСТОМНЫЙ: CV_8U, 64 байта = 512 бит) ---
            cv::Mat desc_a;
            detector_a->compute(frame, corners, desc_a);

            // Проверка, что ваш дескриптор действительно 64 байта (512 бит)
            if (desc_a.cols != 64 || desc_a.type() != CV_8U) {
                std::cerr << "Warning: FAST descriptor is not 64 bytes (cols=" 
                          << desc_a.cols << ", type=" << desc_a.type() << ")\n";
            }

            // --- KAZE-дескрипторы для своих собственных точек ---
            cv::Mat desc_b;
            try
            {
                detector_b->compute(frame, kaze_corners, desc_b);
            }
            catch (const cv::Exception& e)
            {
                std::cerr << "KAZE compute threw: " << e.what() << "\n";
                continue;
            }

            if (desc_b.empty() || desc_b.rows < 2)
            {
                std::cout << "KAZE produced no descriptors\n";
                continue;
            }

            // --- проверка на NaN/Inf ---
            const bool has_nan = !cv::checkRange(desc_b, true, nullptr, 0, 0);
            double mn = 0.0, mx = 0.0;
            if (!has_nan)
                cv::minMaxLoc(desc_b, &mn, &mx);

            std::cout << "desc_b type=" << desc_b.type()
                      << " size=" << desc_b.rows << "x" << desc_b.cols;
            if (has_nan)
                std::cout << "  [NaN/Inf present]\n";
            else
                std::cout << "  range=[" << mn << ", " << mx << "]\n";

            if (has_nan)
            {
                std::cout << "Warning: KAZE descriptor contains NaN. Cleaning up...\n";
                cv::Mat mask;
                cv::compare(desc_b, desc_b, mask, cv::CMP_EQ); // NaN != NaN
                desc_b.setTo(0, ~mask);
                
                cv::minMaxLoc(desc_b, &mn, &mx);
                std::cout << "  Cleaned range=[" << mn << ", " << mx << "]\n";
            }

            if (std::abs(mx - mn) < 1e-9)
            {
                std::cout << "KAZE descriptor is constant — MLDB degenerate\n";
                continue;
            }

            // --- перевод CV_32F → CV_8U для KAZE ---
            cv::Mat desc_b_bin;
            if (desc_b.type() == CV_8U)
            {
                desc_b_bin = desc_b;
            }
            else if (mx > 1.5)
            {
                desc_b.convertTo(desc_b_bin, CV_8U);
            }
            else
            {
                std::vector<float> flat(desc_b.begin<float>(), desc_b.end<float>());
                std::nth_element(flat.begin(),
                                 flat.begin() + flat.size() / 2, flat.end());
                const float thr = flat[flat.size() / 2];
                std::cout << "KAZE normalized, median threshold=" << thr << "\n";

                desc_b_bin = cv::Mat::zeros(desc_b.size(), CV_8U);
                for (int i = 0; i < desc_b.rows; ++i)
                    for (int j = 0; j < desc_b.cols; ++j)
                        if (desc_b.at<float>(i, j) > thr)
                            desc_b_bin.at<uchar>(i, j) = 255;
            }

            // Универсальная функция построения гистограммы Хэмминга
            auto draw_hamming_hist = [](const cv::Mat& desc,
                                        const std::string& title,
                                        cv::Scalar bar_color) -> cv::Mat
            {
                const int W = 640, H = 400, B = 64; // 64 столбца (бина) для гистограммы
                cv::Mat img(H, W, CV_8UC3, cv::Scalar(40, 40, 40));

                if (desc.empty() || desc.rows < 2 || desc.cols == 0 || desc.type() != CV_8U)
                {
                    cv::putText(img, title + ": bad descriptor",
                                {10, 22}, cv::FONT_HERSHEY_SIMPLEX,
                                0.6, {255, 255, 255}, 1);
                    return img;
                }

                std::vector<float> d;
                d.reserve(desc.rows * (desc.rows - 1) / 2);
                
                // Вычисление расстояния Хэмминга для всех пар
                for (int i = 0; i < desc.rows; ++i) {
                    for (int j = i + 1; j < desc.rows; ++j) {
                        int hd = 0;
                        for (int k = 0; k < desc.cols; ++k) {
                            hd += __builtin_popcount(
                                      static_cast<unsigned>(
                                          desc.at<uchar>(i, k) ^ desc.at<uchar>(j, k)));
                        }
                        d.push_back(static_cast<float>(hd));
                    }
                }

                // <--- КЛЮЧЕВОЙ МОМЕНТ: Динамический расчет максимального числа бит
                const int max_bits = desc.cols * 8; 
                const float range[]   = {0.f, static_cast<float>(max_bits + 4)};
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
                        const int h = static_cast<int>(hist.at<float>(i) / max_h * (H - 60));
                        cv::rectangle(img,
                                      {i * bin_w, H - h},
                                      {(i + 1) * bin_w - 1, H - 1},
                                      bar_color, cv::FILLED);
                    }
                }

                cv::putText(img, title, {10, 22},
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1);
                cv::putText(img,
                            "pairs: " + std::to_string(d.size()) +
                            "   bits: " + std::to_string(max_bits) +
                            "   max: "  + std::to_string(static_cast<int>(max_h)),
                            {10, H - 10}, cv::FONT_HERSHEY_SIMPLEX,
                            0.5, {200, 200, 200}, 1);
                return img;
            };

            // Передаем оба дескриптора. Функция сама определит, что cols=64, 
            // и построит гистограмму для 512 бит (диапазон 0..516, 64 бина).
            cv::Mat h_a = draw_hamming_hist(desc_a,     "FAST (512 bit)", {0, 255, 0});
            cv::Mat h_b = draw_hamming_hist(desc_b_bin, "KAZE (512 bit)", {0, 180, 255});

            cv::Mat both;
            cv::hconcat(h_a, h_b, both);
            cv::imshow("Hamming histograms", both);
        }

        std::cout << "FAST: " << corners.size() 
                  << "  KAZE: " << kaze_corners.size() << "\r" << std::flush;
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}