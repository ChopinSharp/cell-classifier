#pragma once

#include <opencv2/opencv.hpp>
#include <CellClassifier.h>
#include <CellSegmenter.h>
#include <QPixmap>
#include <QObject>

using cv::Mat;
using std::string;

class CellProcessor : public QObject
{
	Q_OBJECT

public:
	CellProcessor(string cls_model_url, string seg_model_url, float saturation=0.0035): 
		classifier(cls_model_url), segmenter(seg_model_url), saturation(saturation), has_image(false) {}
	void load_image(string image_url, bool verbose=true);
	const Mat &get_original_image() { return original_image; }
	const Mat &get_enhanced_image() { return enhanced_image; }
	const Mat &get_normalized_image() { return normalized_image; }
	shared_ptr<Pred> predict_single(const Roi &roi = Roi()) { return classifier.predict_single(normalized_image, roi); };

	/* Slots and signals for classifier */
public slots:
	void predict_batch(QString q_folder_url);
	void save_results_to_file(QString q_file_name);
signals:
	void update_progress_bar(int progress);
	void update_status_bar(QString info);
	void cls_result_ready();

	/* Slots and signals for segmentor */
public slots:
	void start_infer();
signals:
	void seg_result_ready(QPixmap result);

private:
	string image_url;
	float saturation;
	Mat original_image;
	Mat enhanced_image;
	Mat normalized_image;
	CellClassifier classifier;
	CellSegmenter segmenter;
	shared_ptr<std::vector<NamedPred>> batch_results;

public:
	bool has_image;
};