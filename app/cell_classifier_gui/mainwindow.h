#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsPixmapItem>
#include "CellClassifier.h"
#include <QProgressBar>
#include <QtCharts>
QT_CHARTS_USE_NAMESPACE

namespace Ui {
	class MainWindow;
}

class BatchPredictor : public QObject
{
	Q_OBJECT

public:
	BatchPredictor() : results(nullptr) {}

public slots:
	void when_start_batch_predicton(CellClassifier classifier, QString q_folder_url);
	void when_save_result_to_file(QString q_file_name);

signals:
	void update_progress_bar(int progress);
	void update_status_bar(QString info);
	void result_ready();

private:
	shared_ptr<std::vector<NamedPred>> results;
};

/*
class Segmenter : public QObject
{

};
*/

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
	void predict_batch();

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
	void when_result_ready();

signals:
	void prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf);
	void start_batch_prediction(CellClassifier classifier, QString q_folder_url);
	void save_result_to_file(QString file_name);

private:
    Ui::MainWindow *ui;
    QGraphicsPixmapItem *img;
    QGraphicsScene *scene;
    QRect sl_rect;
    QGraphicsRectItem *rect_item = nullptr;
	QString file_name;
	QChart *chart;
	QProgressBar *progress_bar;
    bool has_image = false;
	CellClassifier classifier;
	QThread batch_thread;
	BatchPredictor batch_predictor;
};



#endif // MAINWINDOW_H
