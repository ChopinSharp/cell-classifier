#pragma once

#include <memory>
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using std::shared_ptr;
using cv::Mat;
using cv::Vec3b;

class CellSegmenter
{
public:
	CellSegmenter() {}
	CellSegmenter(string model_url, bool verbose = false);
	Mat infer(const Mat &image);

private:
	shared_ptr<torch::jit::script::Module> module;

public:
	static const Vec3b palette[4];
};