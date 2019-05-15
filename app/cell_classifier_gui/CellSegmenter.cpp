#include "CellSegmenter.h"
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

const Vec3b CellSegmenter::palette[]{ {0, 0, 0}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255} };

CellSegmenter::CellSegmenter(string model_url, bool verbose)
{
	/* Load model */
	if (!fs::exists(model_url))
	{
		cerr << "FATAL ERROR: CellSegmenter::CellSegmenter: Model not found, exiting ..." << endl;
		system("pause");
		exit(1);
	}
	this->module = torch::jit::load(model_url);
	if (this->module == nullptr)
	{
		cerr << "FATAL ERROR: CellSegmenter::CellSegmenter: Fail to load model at " << model_url << ", exiting ..." << endl;
		system("pause");
		exit(2);
	}
}

Mat CellSegmenter::infer(const Mat& image)
{
	/* Resize to a multiple of 32 */
	Mat resized_image;
	resize(image, resized_image, Size(image.size[1] / 32 *32, image.size[0] / 32 * 32));
	Mat mean_mat, std_mat;
	meanStdDev(resized_image, mean_mat, std_mat);
	auto mean = mean_mat.at<double>(0, 0), std = std_mat.at<double>(0, 0);
	torch::Tensor input = torch::from_blob(resized_image.data, { 1, resized_image.rows, resized_image.cols, 3 }, torch::kFloat32)
		.permute({ 0, 3, 1, 2 }).toType(torch::kFloat).sub(mean).div(std);

	/* Execute the forward pass of the model */
	auto heat_maps = (this->module)->forward({ input }).toTensor(); // std::vector<torch::jit::IValue> inputs{tensor_image};
	auto pred_map = heat_maps.squeeze().argmax(0);
	auto pred_map_accessor = pred_map.accessor<int64_t, 2>();
	Mat color_map(pred_map.size(0), pred_map.size(1), CV_8SC3);
	for (int i = 0; i < pred_map.size(0); i++)
	{
		for (int j = 0; j < pred_map.size(1); j++)
		{
			color_map.at<Vec3b>(i, j) = palette[pred_map_accessor[i][j]];
		}
	}
	Mat result;
	resize(color_map, result, Size(image.size[1], image.size[0]), 0., 0., INTER_NEAREST);
	return result;
}
