#include "CellProcessor.h"
#include "utils.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <QImage>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

/* Load image and fill in original_image, enhanced_image and normalized_image fileds */
void CellProcessor::load_image(string image_url, bool verbose)
{
	/* Read in image */
	Mat ori_image = imread(image_url, IMREAD_GRAYSCALE | IMREAD_ANYDEPTH);
	if (ori_image.data == nullptr)
	{
		cerr << "ERROR: Fail to open image: " << image_url << endl;
		return;
	}

	/* Print out image info if verbose is on */
	if (verbose)
	{
		show_image_info(image_url, ori_image);
	}

	this->image_url = image_url;
	this->has_image = true;

	/* Check image depth, fill in original_image for display */
	if (ori_image.depth() == CV_8U)
	{
		original_image = ori_image;
	}
	else if (ori_image.depth() == CV_16U)
	{
		Mat scaled_image = ori_image / 256;
		scaled_image.convertTo(original_image, CV_8U);
	}
	else
	{
		cerr << "ERROR: Image with unsupported depth: " << image_url << endl;
		return;
	}

	/* Enhance image based on histogram */
	enhanced_image = enhance_image(ori_image, saturation);

	/* Pad and normalize enhanced image for further processing */
	Mat padded_image;
	cvtColor(enhanced_image, padded_image, COLOR_GRAY2BGR);
	padded_image.convertTo(normalized_image, CV_32F, 1. / 255.);
}


void CellProcessor::predict_batch(QString q_folder_url)
{
	string folder_url = q_folder_url.toStdString();
	cerr << "In CellProcessor::predict_batch" << endl;
	batch_results = std::make_shared<std::vector<NamedPred>>();

	/* Get base directory length, for URL formatting */
	auto base_length = folder_url.length();
	if (folder_url[base_length - 1] != '\\')
	{
		base_length += 1;
	}

	/* Iterate and Infer */
	int progress = 1;
	for (auto& p : fs::recursive_directory_iterator(folder_url))
	{
		if (fs::is_regular_file(p.path()))
		{
			auto relative_url = p.path().string().substr(base_length);
			emit update_status_bar(QString::fromStdString("Running batch inference - processing \"" + relative_url + "\""));

			auto pred = classifier.predict_single(p.path().string(), saturation);
			if (pred != nullptr)
			{
				NamedPred result;
				result.first = relative_url;
				result.second = pred;
				this->batch_results->push_back(result);
			}
			else
			{
				cerr << "Error: CellProcessor::predict_batch: Fail to classify " << p.path().string() << endl;
			}
		}
		emit update_progress_bar(++progress);
	}
	emit update_status_bar(QString("Batch inference complete."));
	emit cls_result_ready();
}

void CellProcessor::save_results_to_file(QString q_file_name)
{
	cerr << "In BatchPredictor::on_save_result_to_file";
	if (q_file_name.isNull())
	{
		emit update_status_bar(QString("Results unsaved, canceled by user."));
		return;
	}
	string file_name = q_file_name.toStdString();
	ofstream fout(file_name);
	int count[3]{ 0, 0, 0 };
	
	/* CSV Header */
	fout << "File, Prediction, Fragmented, Hyperfused, WT\n";
	
	/* Content */
	for (const auto &iter : *batch_results)
	{
		count[(iter.second)->first]++;
		fout << iter.first << ", " << CellClassifier::class_names[(iter.second)->first];
		for (int i = 0; i < 3; i++)
		{
			fout << ", " << (iter.second)->second[i];
		}
		fout << "\n";
	}
	/* Statistics */
	auto total = count[0] + count[1] + count[2];
	fout << "\"[ CONCLUSION ] " << total << " images in total";
	for (int i = 0; i < 3; i++)
	{
		fout << ", " << count[i] << " " << CellClassifier::class_names[i]
			<< " (" << 100 * count[i] / total << "%)";
	}
	fout << ".\"\n";
	fout.close();
	emit update_status_bar(QString::fromStdString("Results saved to \"" + file_name + "\""));
}

void CellProcessor::start_infer()
{
	Mat result = segmenter.infer(this->normalized_image);
	emit seg_result_ready(QPixmap::fromImage(QImage(result.data, result.cols, result.rows, QImage::Format_RGB888)));
}

