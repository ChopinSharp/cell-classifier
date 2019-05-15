#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QDebug>
#include <QScrollBar>
#include <QFont>
#include <filesystem>
#include <memory>
#include <fstream>
#include <QMetaType>


namespace fs = std::filesystem;

QChart *create_bar_chart(double fg_conf, double hf_conf, double wt_conf)
{
	QChart *chart = new QChart();
	QHorizontalBarSeries *series = new QHorizontalBarSeries(chart);

	QBarSet *set = new QBarSet("", series);
	//QBarSet *L_pad = new QBarSet("", series);
	//QBarSet *R_pad = new QBarSet("", series);
	*set << wt_conf << hf_conf << fg_conf;
	//*L_pad << 0 << 0 << 0;
	//*R_pad << 0 << 0 << 0;
	//series->append(L_pad);
	series->append(set);
	//series->append(R_pad);

	chart->addSeries(series);
	chart->setTitle("Bar Chart of Confidence");
	chart->setAnimationOptions(QChart::SeriesAnimations);

	QStringList categories;
	categories << "WT" << "Hyperfused" << "Fragmented";
	QBarCategoryAxis *axis = new QBarCategoryAxis(chart);
	axis->append(categories);
	chart->createDefaultAxes();
	chart->setAxisY(axis, series);
	chart->axisX()->setMax(QVariant::fromValue(1));
	chart->legend()->hide();
	//chart->setFont(QFont("Microsoft YaHei UI", 12));
	chart->setTitleFont(QFont("Microsoft YaHei UI", 14, QFont::Bold));
	return chart;
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
	processor(
		R"(models\classification\squeezenet%Sun Mar 31 17#59#37 2019%0.677%0.12210%0.14149.pt)",
		R"(models\segmentation\UNetVggVar0512.pt)"
	)
{
    ui->setupUi(this);

	/* Setup graphics view and chart view */
    this->scene = new QGraphicsScene(this);
    this->img = scene->addPixmap(QPixmap());
	this->chart = new QChart();
    ui->graphicsView->setScene(scene);
    ui->graphicsView->setDragMode(QGraphicsView::RubberBandDrag);
    ui->graphicsView->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    ui->graphicsView->setBackgroundRole(QPalette::Midlight);
	ui->chartView->setRenderHint(QPainter::Antialiasing);
	
	/* Set window title */
	setWindowTitle("Cell Classifier GUI");

	/* Setup progress bar */
	progress_bar = new QProgressBar();
	progress_bar->setFixedWidth(120);
	progress_bar->hide();

	/* Setup status bar */
	ui->statusBar->addPermanentWidget(progress_bar);
	ui->statusBar->showMessage(QString("Ready."));

	/* Setup radio button */
	ui->radioButtonOri->setEnabled(false);
	ui->radioButtonEnh->setEnabled(false);
	ui->radioButtonSeg->setEnabled(false);


	/* Setup thread for batch inference */
	processor.moveToThread(&thread);
	thread.start();

	/* Connect signals and slots */
	/* sync prediction result with chart and lables */
	connect(this, &MainWindow::prediction_changed, this, &MainWindow::label_on_prediction_changed);
	connect(this, &MainWindow::prediction_changed, this, &MainWindow::chart_on_prediction_changed);
	/* batch prediction */
	connect(&processor, &CellProcessor::update_progress_bar, this, &MainWindow::when_update_progress_bar);
	connect(&processor, &CellProcessor::update_status_bar,   this, &MainWindow::when_update_status_bar);
	connect(&processor, &CellProcessor::cls_result_ready,    this, &MainWindow::when_cls_result_ready);
	connect(this, &MainWindow::save_results_to_file, &processor, &CellProcessor::save_results_to_file);
	connect(this, &MainWindow::start_batch_predict,  &processor, &CellProcessor::predict_batch);
	/* segmentation */
	connect(this, &MainWindow::start_infer, &processor, &CellProcessor::start_infer);
	connect(&processor, &CellProcessor::seg_result_ready, this, &MainWindow::when_seg_result_ready);
}

MainWindow::~MainWindow()
{
    delete ui;
	thread.quit();
	thread.wait();
}

void MainWindow::on_actionOpenImage_triggered()
{
	/* Read in file name */
	auto file_name = QFileDialog::getOpenFileName(
		this, tr("Open Image"), "", tr("Image Files (*.tif *.tiff *.jpg)")
	);
	if (file_name.isNull())
	{
		ui->statusBar->showMessage(QString("No image opened, canceled by user."));
		return;
	}
	setWindowTitle(file_name + " - Cell Classifier GUI");
	qDebug() << "file selected: " << file_name;

	/* Disable radio buttons */
	ui->radioButtonOri->setEnabled(false);
	ui->radioButtonEnh->setEnabled(false);
	ui->radioButtonSeg->setEnabled(false);
	ui->menuFile->setEnabled(false);

	/* Remove old roi box */
	if (rect_item != nullptr)
	{
		this->scene->removeItem(this->rect_item);
		delete rect_item;
		rect_item = nullptr;
	}

	/* Load image with processor object */
	this->processor.load_image(file_name.toStdString());

	/* Construct QPixmap objects and enable radio buttons */
	auto original_mat = this->processor.get_original_image();
	this->original_pixmap = QPixmap::fromImage(QImage(original_mat.data, original_mat.cols, original_mat.rows, QImage::Format_Grayscale8));
	auto enhanced_mat = this->processor.get_enhanced_image();
	this->enhanced_pixmap = QPixmap::fromImage(QImage(enhanced_mat.data, enhanced_mat.cols, enhanced_mat.rows, QImage::Format_Grayscale8));
	ui->radioButtonOri->setEnabled(true);
	ui->radioButtonEnh->setEnabled(true);

	/* Display orignal image by default */
	ui->radioButtonOri->setChecked(true);
	this->img->setPixmap(original_pixmap);
    this->scene->setSceneRect(0, 0, img->pixmap().width(), img->pixmap().height());
	const QString message = tr("Opened \"%1\", %2x%3, Depth: %4")
		.arg(QDir::toNativeSeparators(file_name))
		.arg(img->pixmap().width()).arg(img->pixmap().height()).arg(img->pixmap().depth());
	ui->statusBar->showMessage(message);

	/* Do classification */
	auto pred = this->processor.predict_single();
	if (pred != nullptr)
	{
		/* update UI */
		double fg_conf = pred->second[0], hf_conf = pred->second[1], wt_conf = pred->second[2];
		emit prediction_changed(pred->first, fg_conf, hf_conf, wt_conf);
	}
	else
	{
		ui->statusBar->showMessage("Fail to infer on \"" + file_name + "\".");
	}

	/* Do segmentation */
	emit start_infer();
}

void MainWindow::on_actionOpenFolder_triggered()
{
	qDebug() << "in MainWindow::on_actionOpenFolder_triggered";
	ui->menuFile->setEnabled(false);
	QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"));
	if (dir.isNull())
	{
		ui->statusBar->showMessage(QString("Batch inference aborted."));
		return;
	}
	qDebug() << "folder selected: " << dir;
	string folder_url = dir.toStdString();

	int total = 0;
	for (auto& p : fs::recursive_directory_iterator(folder_url)) total++;

	/* Prepare progress bar */
	progress_bar->setMaximum(total);
	progress_bar->setMinimum(0);
	progress_bar->setValue(0);
	progress_bar->show();

	emit start_batch_predict(QString::fromStdString(folder_url));
}

void MainWindow::on_radioButtonOri_clicked()
{
	qDebug() << "show original";
	this->img->setPixmap(this->original_pixmap);
}

void MainWindow::on_radioButtonEnh_clicked()
{
	qDebug() << "show enhanced";
	this->img->setPixmap(this->enhanced_pixmap);
}

void MainWindow::on_radioButtonSeg_clicked()
{
	qDebug() << "show segmented";
	this->img->setPixmap(this->segmented_pixmap);
}

void MainWindow::on_graphicsView_rubberBandChanged(const QRect &viewportRect, const QPointF &fromScenePoint, const QPointF &toScenePoint)
{
    /* no image loaded */
    if (!(this->processor.has_image)) return ;

    /* selection done */
    if (viewportRect.isNull())
    {
        /* rectify selction area */
		int new_top  = sl_rect.top()  + ui->graphicsView->verticalScrollBar()->value();
		int new_left = sl_rect.left() + ui->graphicsView->horizontalScrollBar()->value();
		sl_rect.moveTo(new_left, new_top);
        auto img_height = img->pixmap().height();
        auto img_width  = img->pixmap().width();
        if (sl_rect.bottom() >= img_height)
        {
            sl_rect.setBottom(img_height - 1);
        }
        if (sl_rect.right() >= img_width)
        {
            sl_rect.setRight(img_width - 1);
        }
        if (sl_rect.left() < 0)
        {
            sl_rect.setLeft(0);
        }
        if (sl_rect.top() < 0)
        {
            sl_rect.setTop(0);
        }
        qDebug() << sl_rect;
		ui->statusBar->showMessage(
			tr("Infer on area: (%1,%2) %3x%4.")
			.arg(sl_rect.left())
			.arg(sl_rect.top())
			.arg(sl_rect.width())
			.arg(sl_rect.height())
		);

		/* delete old selection rect */
        if (rect_item != nullptr)
        {
            this->scene->removeItem(this->rect_item);
            delete rect_item;
			rect_item = nullptr;
        }

		/* draw new selection rect */
        this->rect_item = this->scene->addRect(this->sl_rect, QPen(Qt::yellow));

		/* make prediction to selection area */
		auto pred = this->processor.predict_single(Roi(sl_rect.top(), sl_rect.bottom(), sl_rect.left(), sl_rect.right()));
		double fg_conf = pred->second[0], hf_conf = pred->second[1], wt_conf = pred->second[2];
		
		/* update UI */
		emit prediction_changed(pred->first, fg_conf, hf_conf, wt_conf);
    }

    /* during selection */
    else
    {
        this->sl_rect = viewportRect;
		ui->statusBar->showMessage(
			tr("Selected area: (%1, %2), %3 x %4.")
			.arg(viewportRect.left())
			.arg(viewportRect.top())
			.arg(viewportRect.width())
			.arg(viewportRect.height())
		);
    }
}

void MainWindow::label_on_prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf)
{
	ui->pred->setText(QString::fromStdString(CellClassifier::class_names[pred]));
	ui->fg_conf->setText(QString::number(fg_conf));
	ui->hf_conf->setText(QString::number(hf_conf));
	ui->wt_conf->setText(QString::number(wt_conf));
}

void MainWindow::chart_on_prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf)
{
	/* delete operation must be after the call to setChart ~ */
	ui->chartView->setChart(create_bar_chart(fg_conf, hf_conf, wt_conf));
	delete this->chart;
	this->chart = ui->chartView->chart();
}

void MainWindow::when_update_progress_bar(int progress)
{
	qDebug() << "in MainWindow::on_update_progress_bar";
	progress_bar->setValue(progress);
}

void MainWindow::when_update_status_bar(QString info)
{
	qDebug() << "in MainWindow::on_update_status_bar";
	ui->statusBar->showMessage(info);
}

void MainWindow::when_cls_result_ready()
{
	qDebug() << "in MainWindow::on_result_ready";
	QString file_name = QFileDialog::getSaveFileName(this, tr("Save to"), "", tr("CSV File (*.csv)"));
	qDebug() << file_name;
	emit save_results_to_file(file_name);
	progress_bar->hide();
	ui->menuFile->setEnabled(true);
}

void MainWindow::when_seg_result_ready(QPixmap result)
{
	this->segmented_pixmap = result;
	ui->radioButtonSeg->setEnabled(true);
	ui->menuFile->setEnabled(true);
}
