#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsPixmapItem>
#include <QProgressBar>
#include <QtCharts>
#include "CellProcessor.h"
QT_CHARTS_USE_NAMESPACE

namespace Ui {
	class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_actionOpenImage_triggered();
	void on_actionOpenFolder_triggered();
	void on_pushButtonSeg_clicked();
	void on_radioButtonOri_clicked();
	void on_radioButtonEnh_clicked();
	void on_radioButtonSeg_clicked();
	void on_radioButtonWT_clicked();
	void on_radioButtonFg_clicked();
	void on_radioButtonHf_clicked();
	void on_fg_enabled_clicked();
	void on_hf_enabled_clicked();
	void on_wt_enabled_clicked();
    void on_graphicsView_rubberBandChanged(const QRect &viewportRect, const QPointF &fromScenePoint, const QPointF &toScenePoint);
	void table_on_prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf);
	void chart_on_prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf);
	void chart_on_proportion_changed(double fg, double hf, double wt);
	void when_update_progress_bar(int progress);
	void when_update_status_bar(QString info);
	void when_cls_result_ready();
	void when_seg_result_ready();
	void when_wt_heat_map_ready(QPixmap map);
	void when_fg_heat_map_ready(QPixmap map);
	void when_hf_heat_map_ready(QPixmap map);

signals:
	void prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf);
	void proportion_changed(double fg, double hf, double wt);
	void start_batch_predict(QString q_folder_url);
	void save_results_to_file(QString q_file_name);
	void start_infer(bool fg, bool hf, bool wt);

private:
    Ui::MainWindow *ui;
	CellProcessor processor;
	QThread thread;
    QGraphicsPixmapItem *img;
    QGraphicsScene *scene;
    QRect sl_rect;
    QGraphicsRectItem *rect_item = nullptr;
	QProgressBar *progress_bar;
	QTableWidgetItem *cls_pred;
	QTableWidgetItem *wt_prob;
	QTableWidgetItem *fg_prob;
	QTableWidgetItem *hf_prob;
	QPixmap original_pixmap;
	QPixmap enhanced_pixmap;
	QPixmap segmented_pixmap;
	QPixmap wt_heatmap;
	QPixmap fg_heatmap;
	QPixmap hf_heatmap;
	Roi current_roi;
	void update_segmentation_pixmap_and_proportion_chart();
};

#endif // MAINWINDOW_H
