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
	classifier(R"(models\classification\squeezenet%Sun Mar 31 17#59#37 2019%0.677%0.12210%0.14149.pt)")
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

	/* Setup thread for batch inference */
	batch_predictor.moveToThread(&batch_thread);
	batch_thread.start();

	/* Connect signals and slots */
	qRegisterMetaType<CellClassifier>("CellClassifier");
	connect(this, &MainWindow::prediction_changed, this, &MainWindow::label_on_prediction_changed);
	connect(this, &MainWindow::prediction_changed, this, &MainWindow::chart_on_prediction_changed);
	connect(&batch_predictor, &BatchPredictor::update_progress_bar, this, &MainWindow::when_update_progress_bar);
	connect(&batch_predictor, &BatchPredictor::update_status_bar, this, &MainWindow::when_update_status_bar);
	connect(&batch_predictor, &BatchPredictor::result_ready, this, &MainWindow::when_result_ready);
	connect(this, &MainWindow::save_result_to_file, &batch_predictor, &BatchPredictor::when_save_result_to_file);
	connect(this, &MainWindow::start_batch_prediction, &batch_predictor, &BatchPredictor::when_start_batch_predicton);

}

MainWindow::~MainWindow()
{
    delete ui;
	batch_thread.quit();
	batch_thread.wait();
}

void MainWindow::on_actionOpenImage_triggered()
{
	/* read in file name */
	auto tmp_str = QFileDialog::getOpenFileName(
		this, tr("Open Image"), "", tr("Image Files (*.tif *.tiff *.jpg)"));
	if (tmp_str.isNull())
	{
		ui->statusBar->showMessage(QString("No image opened, canceled by user."));
		return;
	}
	this->file_name = tmp_str;

	/* remove old roi box */
	if (rect_item != nullptr)
	{
		this->scene->removeItem(this->rect_item);
		delete rect_item;
		rect_item = nullptr;
	}

	/* load and show image */
	setWindowTitle(file_name + " - Cell Classifier GUI");
    qDebug() << "file selected: " << file_name;
    this->img->setPixmap(QPixmap(file_name));
    this->scene->setSceneRect(0, 0, img->pixmap().width(), img->pixmap().height());
    this->has_image = true;
	const QString message = tr("Opened \"%1\", %2x%3, Depth: %4")
		.arg(QDir::toNativeSeparators(file_name))
		.arg(img->pixmap().width()).arg(img->pixmap().height()).arg(img->pixmap().depth());
	ui->statusBar->showMessage(message);

	/* make prediction */
	auto pred = this->classifier.predict_single(file_name.toStdString(), Roi(), true);
	if (pred != nullptr)
	{
		/* update UI */
		double fg_conf = pred->second[0], hf_conf = pred->second[1], wt_conf = pred->second[2];
		emit prediction_changed(pred->first, fg_conf, hf_conf, wt_conf);
	}
	else
	{
		ui->statusBar->showMessage("Fail to infer on \"" + this->file_name + "\".");
	}
}

void MainWindow::on_actionOpenFolder_triggered()
{
	predict_batch();
}

void MainWindow::on_radioButtonOri_clicked()
{
	qDebug() << "show original";

}

void MainWindow::on_radioButtonEnh_clicked()
{
	qDebug() << "show enhanced";

}

void MainWindow::on_radioButtonSeg_clicked()
{
	qDebug() << "show segmentation";
}

void MainWindow::on_graphicsView_rubberBandChanged(const QRect &viewportRect, const QPointF &fromScenePoint, const QPointF &toScenePoint)
{
    /* no image loaded */
    if (!has_image) return ;

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
        // qDebug() << sl_rect;
		// qDebug() << sl_rect.top() << " " << sl_rect.bottom() << " " << sl_rect.left() << " " << sl_rect.right();
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
		auto pred = this->classifier.predict_single(
			file_name.toStdString(),
			Roi(sl_rect.top(), sl_rect.bottom(), sl_rect.left(), sl_rect.right()), 
			true
		);
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

void MainWindow::when_result_ready()
{
	qDebug() << "in MainWindow::on_result_ready";
	QString file_name = QFileDialog::getSaveFileName(this, tr("Save to"), "", tr("CSV File (*.csv)"));
	qDebug() << file_name;
	emit save_result_to_file(file_name);
	progress_bar->hide();

}

void MainWindow::predict_batch()
{
	qDebug() << "in predict_batch";
	QString dir = QFileDialog::getExistingDirectory(this, tr("Open Directory"));
	if (dir.isNull())
	{
		ui->statusBar->showMessage(QString("Batch inference aborted."));
		return ;
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

	emit start_batch_prediction(this->classifier, QString::fromStdString(folder_url));
}

void BatchPredictor::when_start_batch_predicton(CellClassifier classifier, QString q_folder_url)
{
	string folder_url = q_folder_url.toStdString();
	qDebug() << "in BatchPredictor::on_start_batch_predicton";
	this->results = std::make_shared<std::vector<NamedPred>>();

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

		if (fs::is_directory(p.path()))
		{
			// skip directory entry
		}
		else if (fs::is_regular_file(p.path()))
		{
			auto relative_url = p.path().string().substr(base_length);
			emit update_status_bar(QString::fromStdString("Running batch inference - processing \"" + relative_url + "\""));

			auto pred = classifier.predict_single(p.path().string(), Roi(), false);
			if (pred != nullptr)
			{
				NamedPred result;
				result.first = relative_url;
				result.second = pred;
				this->results->push_back(result);
			}
			else
			{
				// TODO ... single inference failed ...
			}
		}
		else
		{
			// unlikely to reach, no action taken ...
		}
		emit update_progress_bar(++progress);
	}
	emit update_status_bar(QString("Batch inference complete."));
	emit result_ready();
}

void BatchPredictor::when_save_result_to_file(QString q_file_name)
{
	qDebug() << "in BatchPredictor::on_save_result_to_file";
	if (q_file_name.isNull())
	{
		emit update_status_bar(QString("Results unsaved, canceled by user."));
		return;
	}
	string file_name = q_file_name.toStdString();
	std::ofstream fout(file_name);
	int count[3]{ 0, 0, 0 };
	/* Header */
	fout << "File, Prediction, Fragmented, Hyperfused, WT\n";
	/* Content */
	for (const auto &iter : *results)
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

