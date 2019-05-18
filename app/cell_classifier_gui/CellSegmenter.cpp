#include "CellSegmenter.h"
#include <filesystem>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

CellSegmenter::CellSegmenter(string folder_url, bool verbose)
{
	string model_url;
	try
	{
		auto dir_iter = fs::directory_iterator(folder_url);
		if (dir_iter == fs::end(dir_iter))
		{
			throw runtime_error("No segmenter model file under " + folder_url);
		}
		model_url = dir_iter->path().string();
	}
	catch (const runtime_error &e)
	{
		cerr << string("Fatal Error: Unable to find segmenter model file: ") + e.what() << endl;
		system("pause");
		exit(1);
	}
	
	/* Load model */
	this->module = torch::jit::load(model_url);
	if (this->module == nullptr)
	{
		cerr << "Fatal Error: Fail to load model at " << model_url << ", exiting ..." << endl;
		system("pause");
		exit(1);
	}
	cout << "[ Segmenter Model Info ]" << endl << "* Loc: " << model_url << endl;
}

Mat CellSegmenter::infer(const Mat& image)
{
	/* Resize to a multiple of 32 */
	Mat resized_image;
	auto scaled_size = Size(image.size[1] / 32 * 32, image.size[0] / 32 * 32);
	resize(image, resized_image, scaled_size);
	Mat mean_mat, std_mat;
	meanStdDev(resized_image, mean_mat, std_mat);
	auto mean = mean_mat.at<double>(0, 0), std = std_mat.at<double>(0, 0);
	torch::Tensor input = torch::from_blob(resized_image.data, { 1, resized_image.rows, resized_image.cols, 3 }, torch::kFloat32)
		.permute({ 0, 3, 1, 2 }).toType(torch::kFloat).sub(mean).div(std);

	/* Execute the forward pass of the model */
	auto heat_maps = (this->module)->forward({ input }).toTensor().squeeze().permute({ 1, 2, 0 });
		
	/* Compute heap maps */
	auto probs_tensor = heat_maps.softmax(2, torch::kFloat32);
	Mat probs_mat_scaled(scaled_size, CV_32FC4, probs_tensor.data_ptr()), probs_mat;
	resize(probs_mat_scaled, probs_mat, Size(image.size[1], image.size[0]));
	vector<Mat> prob_mats(4);
	split(probs_mat * 255, prob_mats);
	Mat prob_wt, prob_fg, prob_hf;
	prob_mats[1].convertTo(prob_wt, CV_8U);
	prob_mats[2].convertTo(prob_fg, CV_8U);
	prob_mats[3].convertTo(prob_hf, CV_8U);
	applyColorMap(prob_wt, heat_map_wt, COLORMAP_JET);
	applyColorMap(prob_fg, heat_map_fg, COLORMAP_JET);
	applyColorMap(prob_hf, heat_map_hf, COLORMAP_JET);
	
	/* Return probability maps */
	return probs_mat;
}
