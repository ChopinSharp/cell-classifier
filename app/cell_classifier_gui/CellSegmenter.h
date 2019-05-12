#pragma once

#include <memory>

class CellSegmenter
{
public:
	CellSegmenter() {}
	CellSegmenter(string model_url, bool verbose = false) {};

private:
	shared_ptr<torch::jit::script::Module> module;
};