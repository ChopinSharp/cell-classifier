#pragma once

#include <opencv2/opencv.hpp>

using cv::Mat;
using std::string;
using cv::Range;

void show_image_info(const string &image_url, const Mat &image);
Mat enhance_image(const Mat &ori_image, float saturation);

struct Roi
{
	Range row_range, col_range;
	Roi() : row_range(Range::all()), col_range(Range::all()) {}
	Roi(int top, int bottom, int left, int right) : row_range(top, bottom + 1), col_range(left, right + 1) {}
};
