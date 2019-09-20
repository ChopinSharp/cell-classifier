#pragma once

#include <memory>
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <QObject>
#include "utils.h"

using std::shared_ptr;
using std::pair;
using std::vector;
using cv::Mat;

typedef pair<int, vector<float>> Pred;
typedef pair<string, shared_ptr<Pred>> NamedPred;

class CellClassifier
{
public:
	CellClassifier(string folder_url, bool verbose=true);
	//CellClassifier(const CellClassifier &cc) : 
	//	module(cc.module), mean(cc.mean), std(cc.std), temperature(cc.temperature) {}
	shared_ptr<Pred> predict_single(const Mat &image, const Roi &roi = Roi());
	shared_ptr<vector<NamedPred>> predict_batch(string folder_url, float saturation=0.0035);
	void run_shell();
	static string repeat_str(const string &str, int times);
	static void print_batch_result_to_console(shared_ptr<vector<NamedPred>> results);
	static void save_batch_result_to_csv(shared_ptr<vector<NamedPred>> results, string file_name);


private:
	shared_ptr<torch::jit::script::Module> module;
	double mean;
	double std;
	double temperature;
	bool check_image_depth(const Mat &image)
	{
		return image.depth() == CV_8U || image.depth() == CV_16U;
	}

public:
	static const string class_names[4];
};
