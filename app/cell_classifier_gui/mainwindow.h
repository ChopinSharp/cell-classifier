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

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
	void predict_batch();
	void save_batch_result_to_csv(shared_ptr<vector<NamedPred>> results, QString q_file_name);

private slots:
    void on_actionOpenImage_triggered();
	void on_actionOpenFolder_triggered();
    void on_graphicsView_rubberBandChanged(const QRect &viewportRect, const QPointF &fromScenePoint, const QPointF &toScenePoint);
	void label_on_prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf);
	void chart_on_prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf);

signals:
	void prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf);

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
};

#endif // MAINWINDOW_H
