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

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
	processor(R"(resource\classification)", R"(resource\segmentation)")
{
    ui->setupUi(this);

	/* Setup graphics view and chart view */
    this->scene = new QGraphicsScene(this);
    this->img = scene->addPixmap(QPixmap());
    ui->graphicsView->setScene(scene);
    ui->graphicsView->setDragMode(QGraphicsView::RubberBandDrag);
    ui->graphicsView->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    ui->graphicsView->setBackgroundRole(QPalette::Midlight);
	ui->chartView->setRenderHint(QPainter::Antialiasing);
	ui->chartView2->setRenderHint(QPainter::Antialiasing);

	/* Setup result tabs */
	ui->tabWidget->setTabText(0, "Classification");
	ui->tabWidget->setTabText(1, "Segmentation");
	ui->tabWidget->setCurrentIndex(0);

	/* Setup color bar for segmentation heat maps */
	ui->color_bar_label->setPixmap(QPixmap(R"(resource\misc\color_bar_jet.png)"));

	/* Setup table for classification result display */
	this->cls_pred = new QTableWidgetItem();
	this->fg_prob = new QTableWidgetItem();
	this->hf_prob = new QTableWidgetItem();
	this->wt_prob = new QTableWidgetItem();
	this->cls_pred->setFont(QFont("Microsoft YaHei UI", 14));
	this->fg_prob->setFont(QFont("Microsoft YaHei UI", 14));
	this->hf_prob->setFont(QFont("Microsoft YaHei UI", 14));
	this->wt_prob->setFont(QFont("Microsoft YaHei UI", 14));
	ui->tableWidget->setItem(0, 1, this->cls_pred);
	ui->tableWidget->setItem(1, 1, this->fg_prob);
	ui->tableWidget->setItem(2, 1, this->hf_prob);
	ui->tableWidget->setItem(3, 1, this->wt_prob);

	/* Setup loader gif*/
	auto loader_gif = new QMovie(R"(resource\misc\loader.gif)");
	loader_gif->start();
	ui->load_label->setMovie(loader_gif);
	ui->load_label->hide();

	/* Set window title */
	setWindowTitle("Mitochondrials GUI");

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
	ui->pushButtonSeg->setEnabled(false);

	/* Setup check box */
	ui->fg_enabled->setChecked(true);
	ui->hf_enabled->setChecked(true);
	ui->wt_enabled->setChecked(true);

	/* Setup thread for batch inference */
	processor.moveToThread(&thread);
	thread.start();

	/* Connect signals and slots */
	/* sync prediction result with chart and lables */
	connect(this, &MainWindow::prediction_changed, this, &MainWindow::table_on_prediction_changed);
	connect(this, &MainWindow::prediction_changed, this, &MainWindow::chart_on_prediction_changed);
	connect(this, &MainWindow::proportion_changed, this, &MainWindow::chart_on_proportion_changed);
	/* batch prediction */
	connect(&processor, &CellProcessor::update_progress_bar, this, &MainWindow::when_update_progress_bar);
	connect(&processor, &CellProcessor::update_status_bar,   this, &MainWindow::when_update_status_bar);
	connect(&processor, &CellProcessor::cls_result_ready,    this, &MainWindow::when_cls_result_ready);
	connect(this, &MainWindow::save_results_to_file, &processor, &CellProcessor::save_results_to_file);
	connect(this, &MainWindow::start_batch_predict,  &processor, &CellProcessor::predict_batch);
	/* segmentation */
	connect(this, &MainWindow::start_infer, &processor, &CellProcessor::start_infer);
	connect(&processor, &CellProcessor::seg_result_ready,  this, &MainWindow::when_seg_result_ready);
	connect(&processor, &CellProcessor::wt_heat_map_ready, this, &MainWindow::when_wt_heat_map_ready);
	connect(&processor, &CellProcessor::fg_heat_map_ready, this, &MainWindow::when_fg_heat_map_ready);
	connect(&processor, &CellProcessor::hf_heat_map_ready, this, &MainWindow::when_hf_heat_map_ready);
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
	ui->radioButtonWT->setEnabled(false);
	ui->radioButtonFg->setEnabled(false);
	ui->radioButtonHf->setEnabled(false);
	ui->pushButtonSeg->setEnabled(true);

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
		ui->tabWidget->setCurrentIndex(0);
	}
	else
	{
		ui->statusBar->showMessage("Fail to infer on \"" + file_name + "\".");
	}
}

void MainWindow::on_actionOpenFolder_triggered()
{
	QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"));
	if (dir.isNull())
	{
		ui->statusBar->showMessage(QString("Batch inference aborted."));
		return;
	}
	qDebug() << "folder selected: " << dir;
	ui->menuFile->setEnabled(false);
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

void MainWindow::on_pushButtonSeg_clicked()
{
	/* Check input size */
	auto rows = processor.get_image_rows(), cols = processor.get_image_cols();
	QMessageBox msgBox;
	if (rows < 32 || cols < 32)
	{
		msgBox.setIcon(QMessageBox::Critical);
		msgBox.setText("Image is too small to do segmentation.");
		msgBox.setInformativeText("Both sides of the image should be over 32 pix. Operation aborted.");
		msgBox.exec();
		return;
	}
	int area = rows * cols;
	if (area > 1024 * 1024)
	{
		msgBox.setIcon(QMessageBox::Critical);
		msgBox.setText("Image is too big to do segmentation.");
		msgBox.setInformativeText("Input image exceeds an area of 1024 by 1204, which could lead to excessive memory and CPU usage. Operation aborted");;
		msgBox.exec();
		return;
	}
	if (area > 512 * 512)
	{
		msgBox.setIcon(QMessageBox::Warning);
		msgBox.setText("Image is relatively big for segmentation.");
		msgBox.setInformativeText("Segmenting image this big will take a long time. Also the memory usage will be massive. Continue?");
		msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::Cancel);
		msgBox.setDefaultButton(QMessageBox::Cancel);
		auto choice = msgBox.exec();
		if (choice == QMessageBox::Cancel)
		{
			return;
		}
	}

	/* Do segmentation */
	ui->menuFile->setEnabled(false);
	ui->pushButtonSeg->setEnabled(false);
	ui->fg_enabled->setEnabled(false);
	ui->hf_enabled->setEnabled(false);
	ui->wt_enabled->setEnabled(false);
	// qDebug() << ui->fg_enabled->isChecked() << ui->hf_enabled->isChecked() << ui->wt_enabled->isChecked();
	emit start_infer(ui->fg_enabled->isChecked(), ui->hf_enabled->isChecked(), ui->wt_enabled->isChecked());
	ui->load_label->show();
	
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

void MainWindow::on_radioButtonWT_clicked()
{
	qDebug() << "show WT heat map";
	this->img->setPixmap(this->wt_heatmap);
}

void MainWindow::on_radioButtonFg_clicked()
{
	qDebug() << "show fragmented heat map";
	this->img->setPixmap(this->fg_heatmap);
}

void MainWindow::on_radioButtonHf_clicked()
{
	qDebug() << "show hyperfused heat map";
	this->img->setPixmap(this->hf_heatmap);
}

void MainWindow::update_segmentation_pixmap_and_proportion_chart()
{
	if (processor.has_seg_result)
	{
		processor.set_segmentation_map(
			ui->fg_enabled->isChecked(), ui->hf_enabled->isChecked(), ui->wt_enabled->isChecked());
		this->segmented_pixmap = processor.get_segmentation_pixmap();
		if (ui->radioButtonSeg->isChecked())
		{
			this->img->setPixmap(this->segmented_pixmap);
		}
		if (ui->fg_enabled->isChecked() || ui->hf_enabled->isChecked() || ui->wt_enabled->isChecked())
		{
			auto proportion = processor.calculate_proportion(
				ui->fg_enabled->isChecked(), ui->hf_enabled->isChecked(), ui->wt_enabled->isChecked(), current_roi);
			emit proportion_changed(proportion.fg, proportion.hf, proportion.wt);
		}
		else
		{
			ui->chartView2->setChart(new QChart());
		}
		
	}
}

void MainWindow::on_fg_enabled_clicked()
{
	qDebug() << "fg_enabled: " << ui->fg_enabled->isChecked();
	update_segmentation_pixmap_and_proportion_chart();
}

void MainWindow::on_hf_enabled_clicked()
{
	qDebug() << "hf_enabled: " << ui->hf_enabled->isChecked();
	update_segmentation_pixmap_and_proportion_chart();
}

void MainWindow::on_wt_enabled_clicked()
{
	qDebug() << "wt_enabled: " << ui->wt_enabled->isChecked();
	update_segmentation_pixmap_and_proportion_chart();
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
		current_roi = Roi(sl_rect.top(), sl_rect.bottom(), sl_rect.left(), sl_rect.right());
		auto pred = this->processor.predict_single(current_roi);
		double fg_conf = pred->second[0], hf_conf = pred->second[1], wt_conf = pred->second[2];
		emit prediction_changed(pred->first, fg_conf, hf_conf, wt_conf);

		/* if segmentation has been done, calculate statistics */
		if (processor.has_seg_result && (ui->fg_enabled->isChecked() || ui->hf_enabled->isChecked() || ui->wt_enabled->isChecked()))
		{
			auto proportion = processor.calculate_proportion(
				ui->fg_enabled->isChecked(), ui->hf_enabled->isChecked(), ui->wt_enabled->isChecked(), current_roi);
			emit proportion_changed(proportion.fg, proportion.hf, proportion.wt);
		}
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

void MainWindow::table_on_prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf)
{
	this->cls_pred->setText(QString::fromStdString(CellClassifier::class_names[pred]));
	this->fg_prob->setText(QString::number(fg_conf, 'g', 5));
	this->hf_prob->setText(QString::number(hf_conf, 'g', 5));
	this->wt_prob->setText(QString::number(wt_conf, 'g', 5));
}

QChart *create_bar_chart(double fg_conf, double hf_conf, double wt_conf)
{
	QChart *chart = new QChart();
	QHorizontalBarSeries *series = new QHorizontalBarSeries(chart);

	QBarSet *set = new QBarSet("", series);
	*set << wt_conf << hf_conf << fg_conf;
	series->append(set);

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
	chart->setTitleFont(QFont("Microsoft YaHei UI", 15, QFont::Bold));
	return chart;
}

void MainWindow::chart_on_prediction_changed(int pred, double fg_conf, double hf_conf, double wt_conf)
{
	/* delete operation must be after the call to setChart ~ */
	auto old_chart = ui->chartView->chart();
	ui->chartView->setChart(create_bar_chart(fg_conf, hf_conf, wt_conf));
	delete old_chart;
}

inline QString format_percentage_info(QString category, double ratio)
{
	return QString("%1: %2%").arg(category).arg(100 * ratio, 0, 'f', 0);
}

QChart *create_pie_chart(double fg, double hf, double wt)
{
	/* Prepare data */
	QPieSeries *series = new QPieSeries();
	qDebug() << "create_pie_chart: " << fg << " " << hf << " " << wt;
	series->append(format_percentage_info("Fragmented", fg), fg);
	series->append(format_percentage_info("Hyperfused", hf), hf);
	series->append(format_percentage_info("WT", wt), wt);
	
	series->slices().at(0)->setBrush(Qt::green);
	series->slices().at(1)->setBrush(Qt::blue);
	series->slices().at(2)->setBrush(Qt::red);

	QChart *chart = new QChart();
	chart->addSeries(series);
	chart->setTitle("Pie Chart of Proportion");
	chart->setTitleFont(QFont("Microsoft YaHei UI", 15, QFont::Bold));
	chart->setAnimationOptions(QChart::SeriesAnimations);
	chart->legend()->setFont(QFont("Microsoft YaHei UI", 10));
	chart->legend()->setAlignment(Qt::AlignBottom);
	return chart;
}

void MainWindow::chart_on_proportion_changed(double fg, double hf, double wt)
{
	/* delete operation must be after the call to setChart ~ */
	qDebug() << "MainWindow::chart_on_proportion_changed: " << fg << " " << hf << " " << wt;
	auto old_chart = ui->chartView2->chart();
	ui->chartView2->setChart(create_pie_chart(fg, hf, wt));
	delete old_chart;
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

void MainWindow::when_seg_result_ready()
{
	ui->load_label->hide();
	processor.set_segmentation_map(ui->fg_enabled->isChecked(), ui->hf_enabled->isChecked(), ui->wt_enabled->isChecked());
	this->segmented_pixmap = processor.get_segmentation_pixmap();
	if (ui->fg_enabled->isChecked() || ui->hf_enabled->isChecked() || ui->wt_enabled->isChecked())
	{
		/* WARNING: when none of the three check boxes is checked, emit the following signal will trigger a CRASH */
		/* WARNING: call to calculate_proportion should be after the call to set_segmentation_map */
		auto proportion = processor.calculate_proportion(ui->fg_enabled->isChecked(), ui->hf_enabled->isChecked(), ui->wt_enabled->isChecked());
		emit proportion_changed(proportion.fg, proportion.hf, proportion.wt);
	}
	ui->radioButtonSeg->setEnabled(true);
	ui->fg_enabled->setEnabled(true);
	ui->hf_enabled->setEnabled(true);
	ui->wt_enabled->setEnabled(true);
	ui->menuFile->setEnabled(true);
	ui->tabWidget->setCurrentIndex(1);
}

void MainWindow::when_wt_heat_map_ready(QPixmap map)
{
	this->wt_heatmap = map;
	ui->radioButtonWT->setEnabled(true);
}

void MainWindow::when_fg_heat_map_ready(QPixmap map)
{
	this->fg_heatmap = map;
	ui->radioButtonFg->setEnabled(true);
}

void MainWindow::when_hf_heat_map_ready(QPixmap map)
{
	this->hf_heatmap = map;
	ui->radioButtonHf->setEnabled(true);
}
