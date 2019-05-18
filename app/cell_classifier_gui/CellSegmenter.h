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
	CellSegmenter(string folder_url, bool verbose = false);
	Mat infer(const Mat &image);
	const Mat &get_heat_map_wt() { return heat_map_wt; }
	const Mat &get_heat_map_fg() { return heat_map_fg; }
	const Mat &get_heat_map_hf() { return heat_map_hf; }

private:
	shared_ptr<torch::jit::script::Module> module;
	Mat heat_map_wt;
	Mat heat_map_fg;
	Mat heat_map_hf;
};