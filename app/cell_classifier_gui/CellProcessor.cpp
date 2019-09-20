#include "CellProcessor.h"
#include "utils.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <QImage>
#include <ctime>


using namespace cv;
using namespace std;
namespace fs = std::filesystem;


const Vec3b CellProcessor::palette[]{ {0, 0, 0}, {255, 0, 0}, {0, 255, 0}, {0, 0, 255} };


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
	this->has_seg_result = false;

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

Mat preprocess_for_batch_inference(string image_url, float saturation)
{
	/* Read in image */
	Mat ori_image = imread(image_url, IMREAD_GRAYSCALE | IMREAD_ANYDEPTH);
	if (ori_image.data == nullptr)
	{
		cerr << "ERROR: Fail to open image: " << image_url << endl;
		return Mat();
	}
	else if (ori_image.depth() != CV_8U)
	{
		cerr << "ERROR: only 8bit images are supported for batch inference." << endl;
		return Mat();
	}

	/* Enhance image based on histogram */
	Mat enhanced_image = enhance_image(ori_image, saturation);

	/* Pad and normalize */
	Mat padded_image, normalized_image;
	cvtColor(enhanced_image, padded_image, COLOR_GRAY2BGR);
	padded_image.convertTo(normalized_image, CV_32F, 1. / 255.);
	return normalized_image;
}

void CellProcessor::predict_batch(QString q_folder_url)
{
	string folder_url = q_folder_url.toStdString();
	cerr << "In CellProcessor::predict_batch" << endl;
	batch_results.clear();

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

			/* preprocess image */
			Mat normalized_image = preprocess_for_batch_inference(p.path().string(), saturation);
			if (normalized_image.empty())
			{
				cerr << "Fail to infer on " << p.path() << endl;
				emit update_progress_bar(++progress);
				continue;
			}

			/* using classifier */
			auto pred = classifier.predict_single(normalized_image);
			if (pred == nullptr)
			{
				cerr << "Error: CellProcessor::predict_batch: Fail to classify " << p.path().string() << endl;
				emit update_progress_bar(++progress);
				continue;
			}

			/* using segmenter */
			probs_mat = segmenter.infer(normalized_image);
			int fg_count = 0, hf_count = 0, wt_count = 0;
			for (int i = 0; i < probs_mat.rows; i++)
			{
				for (int j = 0; j < probs_mat.cols; j++)
				{
					auto pix_scores = probs_mat.at<Vec4f>(i, j);
					auto max_score = pix_scores[0];
					int max_index = 0;
					for (int c = 1; c < 4; c++)
					{
						if (pix_scores[c] > max_score)
						{
							max_score = pix_scores[c];
							max_index = c;
						}
					}
					switch (max_index)
					{
						case 1: wt_count++; break;
						case 2: fg_count++; break;
						case 3: hf_count++; break;
					}
				}
			}
			Seg_Proportion proportion(fg_count, hf_count, wt_count);
			
			batch_results.push_back(Batch_Result_Entry(relative_url, pred, proportion));
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
	fout.precision(3);
	int cls_count[4]{ 0, 0, 0, 0 };
	int seg_count[4]{ 0, 0, 0, 0 };
	
	/* CSV Header */
	fout << "File, Cls-Pred, Fragmented, Hyperfused, WT, Seg-Pred, Fragmented, Hyperfused, WT\n";
	
	/* Content */
	for (const auto &iter : batch_results)
	{
		cls_count[(iter.cls_result)->first]++;
		fout << iter.image_url << ", " << CellClassifier::class_names[(iter.cls_result)->first];
		for (int i = 0; i < 3; i++)
		{
			fout << ", " << (iter.cls_result)->second[i];
		}
		seg_count[iter.seg_result.major]++;
		fout << ", " << CellClassifier::class_names[iter.seg_result.major];
		fout << ", " << iter.seg_result.fg << ", " << iter.seg_result.hf << ", " << iter.seg_result.wt;
		fout << "\n";
	}
	/* Statistics */
	auto total = cls_count[0] + cls_count[1] + cls_count[2];
	fout << "\"[ CONCLUSION ] " << total << " images in total";
	fout << " Cls:";
	for (int i = 0; i < 3; i++)
	{
		fout << " " << cls_count[i] << " " << CellClassifier::class_names[i]
			<< " (" << 100 * cls_count[i] / total << "%)";
	}
	fout << " Seg:";
	for (int i = 0; i < 4; i++)
	{
		fout << " " << seg_count[i] << " " << CellClassifier::class_names[i]
			<< " (" << 100 * seg_count[i] / total << "%)";
	}
	fout << ".\"\n";
	fout.close();
	emit update_status_bar(QString::fromStdString("Results saved to \"" + file_name + "\""));
}

inline QPixmap bgr_mat_to_rgb_pixmap(const Mat &mat)
{
	return QPixmap::fromImage(
		QImage(
			mat.data, mat.cols, mat.rows, QImage::Format_RGB888
		).rgbSwapped()
	);
}

void CellProcessor::set_segmentation_map(bool fg, bool hf, bool wt)
{
	cout << "CellProcessor::set_segmentation_map: " << fg << " " << hf << " " << wt << endl;
	seg_map.create(probs_mat.rows, probs_mat.cols, CV_8SC3);
	bool enabled[]{ true, wt, fg, hf };
	for (int i = 0; i < probs_mat.rows; i++)
	{
		for (int j = 0; j < probs_mat.cols; j++)
		{
			auto pix_scores = probs_mat.at<Vec4f>(i, j);
			// 0: bg, 1: wt, 2: fg, 3: hf
			auto max_score = pix_scores[0];
			int max_index = 0;
			for (int c = 1; c < 4; c++)
			{
				if (enabled[c] && pix_scores[c] > max_score)
				{
					max_score = pix_scores[c];
					max_index = c;
				}
			}
			seg_map.at<Vec3b>(i, j) = CellProcessor::palette[max_index];
		}
	}
}

void CellProcessor::start_infer(bool fg, bool hf, bool wt)
{
	probs_mat = segmenter.infer(this->normalized_image);
	set_segmentation_map(fg, hf, wt);
	has_seg_result = true;
	emit seg_result_ready();
	emit wt_heat_map_ready(bgr_mat_to_rgb_pixmap(segmenter.get_heat_map_wt()));
	emit fg_heat_map_ready(bgr_mat_to_rgb_pixmap(segmenter.get_heat_map_fg()));
	emit hf_heat_map_ready(bgr_mat_to_rgb_pixmap(segmenter.get_heat_map_hf()));
}

Seg_Proportion CellProcessor::calculate_proportion(
	bool fg_enabled, bool hf_enabled, bool wt_enabled, const Roi &roi)
{
	// cerr << "CellProcessor::calculate_proportion: " << fg_enabled << " " << hf_enabled << " " << wt_enabled << endl;
	Mat roi_seg = seg_map(roi.row_range, roi.col_range);
	int fg = 0, hf = 0, wt = 0;
	for (int i = 0; i < roi_seg.rows; i++)
	{
		for (int j = 0; j < roi_seg.cols; j++)
		{
			auto pix = roi_seg.at<Vec3b>(i, j);
			if (wt_enabled && pix[0] > 127)
				wt++;
			else if (fg_enabled && pix[1] > 127)
				fg++;
			else if (hf_enabled && pix[2] > 127)
				hf++;
		}
	}
	return Seg_Proportion(fg, hf, wt);
}
