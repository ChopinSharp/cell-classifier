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
	/*CellSegmenter(const CellSegmenter &cs) : module(cs.module), 
		heat_map_wt(cs.heat_map_wt), heat_map_fg(cs.get_heat_map_fg), heat_map_hf(cs.get_heat_map_hf) {}*/
	Mat infer(const Mat &image);
	const Mat &get_heat_map_wt() { return heat_map_wt; }
	const Mat &get_heat_map_fg() { return heat_map_fg; }
	const Mat &get_heat_map_hf() { return heat_map_hf; }

private:
	shared_ptr<torch::jit::script::Module> module;
	/* Cached result as a matter of expedience, DO NOT REUSE IN ANY CASE */
	Mat heat_map_wt;
	Mat heat_map_fg;
	Mat heat_map_hf;
};