#pragma once

#include <opencv2/opencv.hpp>

using cv::Mat;
using std::string;

void show_image_info(const string &image_url, const Mat &image);
Mat enhance_image(const Mat &ori_image, float saturation);
