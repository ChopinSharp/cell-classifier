#pragma once

#include "CellClassifier.h"
#include "CellSegmenter.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <QPixmap>
#include <QObject>

using cv::Mat;
using std::string;

struct Seg_Proportion
{
	double fg, hf, wt;
	Seg_Proportion(int fg_c, int hf_c, int wt_c)
	{
		double total = fg_c + hf_c + wt_c;
		fg = fg_c / total;
		hf = hf_c / total;
		wt = wt_c / total;
	}
};

class CellProcessor : public QObject
{
	Q_OBJECT

public:
	CellProcessor(string cls_model_folder, string seg_model_folder, float saturation=0.0035): 
		classifier(cls_model_folder), segmenter(seg_model_folder), saturation(saturation),
		has_image(false), has_seg_result(false) {}
	void load_image(string image_url, bool verbose=true);
	const Mat &get_original_image() { return original_image; }
	const Mat &get_enhanced_image() { return enhanced_image; }
	const Mat &get_normalized_image() { return normalized_image; }
	int get_image_rows() { return original_image.rows; }
	int get_image_cols() { return original_image.cols; }
	shared_ptr<Pred> predict_single(const Roi &roi = Roi()) { return classifier.predict_single(normalized_image, roi); };
	void set_segmentation_map(bool fg, bool hf, bool wt);
	QPixmap get_segmentation_pixmap() {
		return QPixmap::fromImage(QImage(seg_map.data, seg_map.cols, seg_map.rows, QImage::Format_RGB888)); 
	}
	Seg_Proportion calculate_proportion(bool fg_enabled, bool hf_enabled, bool wt_enabled, const Roi &roi = Roi());

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
	void start_infer(bool fg, bool hf, bool wt);
signals:
	void seg_result_ready();
	void wt_heat_map_ready(QPixmap map);
	void fg_heat_map_ready(QPixmap map);
	void hf_heat_map_ready(QPixmap map);

private:
	string image_url;
	float saturation;
	Mat original_image;
	Mat enhanced_image;
	Mat normalized_image;
	Mat probs_mat;
	Mat seg_map; // depends on scores_mat and category enable configs
	CellClassifier classifier;
	CellSegmenter segmenter;
	shared_ptr<std::vector<NamedPred>> batch_results;

public:
	bool has_image;
	bool has_seg_result;

public:
	static const Vec3b palette[4];
	
};
