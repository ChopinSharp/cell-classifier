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
	void on_radioButtonOri_clicked();
	void on_radioButtonEnh_clicked();
	void on_radioButtonSeg_clicked();
    void on_graphicsView_rubberBandChanged(const QRect &viewportRect, const QPointF &fromScenePoint, const QPointF &toScenePoint);
	void label_on_prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf);
	void chart_on_prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf);
	void when_update_progress_bar(int progress);
	void when_update_status_bar(QString info);
	void when_cls_result_ready();
	void when_seg_result_ready(QPixmap result);

signals:
	void prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf);
	void start_batch_predict(QString q_folder_url);
	void save_results_to_file(QString q_file_name);
	void start_infer();

private:
    Ui::MainWindow *ui;
	CellProcessor processor;
	QThread thread;
    QGraphicsPixmapItem *img;
    QGraphicsScene *scene;
    QRect sl_rect;
    QGraphicsRectItem *rect_item = nullptr;
	QChart *chart;
	QProgressBar *progress_bar;
	QPixmap original_pixmap;
	QPixmap enhanced_pixmap;
	QPixmap segmented_pixmap;
};



#endif // MAINWINDOW_H
