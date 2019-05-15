#include "utils.h"
#include <iostream>

using std::cout;
using std::endl;

void show_image_info(const string &image_url, const Mat &image)
{
	const string depth_str[] = { "CV_8U", "CV_8S", "CV_16U", "CV_16S", "CV_32S", "CV_32F", "CV_64F" };
	cout << "- - - - - - - - - - - - - - - Image  Info - - - - - - - - - - - - - - -" << endl;
	cout << "URL: " << image_url << endl;
	cout << "rows: " << image.rows << " cols: " << image.cols << endl;
	cout << "depth: " << depth_str[image.depth()] << endl;
	cout << "channels:  " << image.channels() << endl;
	cout << "element size: " << image.elemSize() << endl;
	cout << "cvMat type:" << image.type() << endl;
	cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << endl << endl;
}

Mat enhance_image(const Mat &ori_image, float saturation)
{
	/* calculate histogram */
	double min, max;
	cv::minMaxLoc(ori_image, &min, &max);
	Mat hist;
	int histSize = max - min + 1;
	float range[] = { min - 0.5, max + 0.5 };
	const float *histRange[] = { range };
	int channels[] = { 0 };
	calcHist(&ori_image, 1, channels, Mat(), hist, 1, &histSize, histRange, true);
	hist /= ori_image.rows * ori_image.cols;  // normalize histogram

	/* For debug only */
	float _total = 0;
	for (auto iter = hist.begin<float>(); iter != hist.end<float>(); iter++)
	{
		// cout << " * " << *iter << endl;
		_total += *iter;
	}
	cout << "utils.h: enhance_image: hist sanity check, total: " << _total << endl;

	/* Calculate parameters for enhancement */
	float cur_sat = 0;
	int lo = 0, hi = hist.rows - 1;
	while (cur_sat < saturation)
	{
		auto hist_i = hist.ptr<float>(lo)[0], hist_j = hist.ptr<float>(hi)[0];
		if (hist_i < hist_j)
		{
			cur_sat += hist.ptr<float>(lo++)[0];
		}
		else
		{
			cur_sat += hist.ptr<float>(hi--)[0];
		}
	}
	float low = lo + min, high = hi + min;

	/* Histogram based remapping */
	Mat enhanced_image;
	ori_image.convertTo(enhanced_image, CV_8U, 255 / (high - low), low * 255 / (low - high));
	return enhanced_image;
}