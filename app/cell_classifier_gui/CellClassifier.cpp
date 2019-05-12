﻿#include "CellClassifier.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <filesystem>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;


const int input_size = 224;
const unsigned long err_gap = 800;

const string CellClassifier::class_names[]{ "Fragmented", "Hyperfused", "WT" };

CellClassifier::CellClassifier(string model_url, bool verbose)
{
	/* Load model */
	if (!fs::exists(model_url))
	{
		cerr << "FATAL ERROR: Model not found, exiting ..." << endl;
		system("pause");
		exit(1);
	}
	this->module = torch::jit::load(model_url);
	if (this->module == nullptr)
	{
		cerr << "FATAL ERROR: Fail to load model at " << model_url << ", exiting ..." << endl;
		system("pause");
		exit(2);
	}

	/* Some crappy logic to get the params cause C++ don't have a f**king split */
	string temp_str;
	auto pos_1 = model_url.find('%');
	auto pos_2 = model_url.find('%', pos_1 + 1);
	auto pos_3 = model_url.find('%', pos_2 + 1);
	auto pos_4 = model_url.find('%', pos_3 + 1);
	string time = model_url.substr(pos_1 + 1, pos_2 - pos_1 - 1);
	for (auto &ch : time)
	{
		if (ch == '#') ch = ':';
	}
	this->temperature = stod(model_url.substr(pos_2 + 1));
	this->mean = stod(model_url.substr(pos_3 + 1));
	this->std = stod(model_url.substr(pos_4 + 1));
	if (verbose)
	{
		cout << "[Model Info] Time: " << time << " temperature: " << temperature
			 << " mean: " << mean << " std: " << std << endl;
	}
}

shared_ptr<Pred> CellClassifier::predict_single(string image_url, const Roi &roi, bool verbose)
{
	/* Read in image */
	Mat ori_image = imread(image_url, IMREAD_COLOR | IMREAD_ANYDEPTH);
	if (ori_image.data == nullptr) 
	{
		cerr << "ERROR: Fail to open image: " << image_url << endl;
		return nullptr;
	}

	/* Print out image info if verbose is on */
	if (verbose) 
	{
		const string depth_str[] = { "CV_8U", "CV_8S", "CV_16U", "CV_16S", "CV_32S", "CV_32F", "CV_64F" };
		cout << "- - - - - - - - - - - - - - - Image  Info - - - - - - - - - - - - - - -" << endl;
		cout << "URL: " << image_url << endl;
		cout << "rows: " << ori_image.rows << " cols: " << ori_image.cols << endl;
		cout << "depth: " << depth_str[ori_image.depth()] << endl;
		cout << "channels:  " << ori_image.channels() << endl;
		cout << "element size: " << ori_image.elemSize() << endl;
		cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" << endl << endl;
	}

	/* crop ROI */
	Mat roi_image = ori_image(roi.row_range, roi.col_range);
	// namedWindow("Display window", WINDOW_AUTOSIZE);
	// imshow("Display window", roi_image);
	// waitKey(0);

	/* Convert to float32 */
	Mat float_image;
	roi_image.convertTo(float_image, CV_32F);
	
	/* Resize image */
	Mat resized_image;
	resize(float_image, resized_image, Size(input_size, input_size));

	/* Covert image to torch tensor and preprocess */
	int max_val;
	if (ori_image.depth() == CV_8U) 
	{
		max_val = 255;
	}
	else if (ori_image.depth() == CV_16U) 
	{
		max_val = 65535;
	}
	else 
	{
		cerr << "ERROR: Image with unsupported depth: " << image_url << endl;
		return nullptr;
	}
	torch::Tensor tensor_image =
		torch::from_blob(resized_image.data, { 1, input_size, input_size, 3 }, torch::kFloat32)
		.permute({ 0, 3, 1, 2 })
		.toType(torch::kFloat)
		.div(max_val)
		.sub(this->mean)
		.div(this->std);

	/* Execute the forward pass of the model */
	auto scores = (this->module)->forward({ tensor_image }).toTensor(); // std::vector<torch::jit::IValue> inputs{tensor_image};
	auto probs = scores.div(this->temperature).softmax(1);
	auto probs_accessor = probs.accessor<float, 2>();
	auto stupid_placeholder = probs.argmax(1).toType(torch::kInt);
	auto stupid_accessor = stupid_placeholder.accessor<int, 1>();
	auto pred = stupid_accessor[0];
	auto results = make_shared<Pred>();
	results->first = pred;
	for (int i = 0; i < 3; i++)
	{
		results->second.push_back(probs_accessor[0][i]);
	}
	return results;
}

shared_ptr<vector<NamedPred>> CellClassifier::predict_batch(string folder_url) 
{
	auto results = make_shared<vector<NamedPred>>();

	/* Get base directory length, for URL formatting */
	auto base_length = folder_url.length();
	if (folder_url[base_length - 1] != '\\') 
	{
		base_length += 1;
	}

	/* Iterate and Infer */
	for (auto& p : fs::recursive_directory_iterator(folder_url)) 
	{
		if (fs::is_directory(p.path())) 
		{
			continue;
		}
		else if (fs::is_regular_file(p.path())) 
		{
			NamedPred result;
			cout << " - processing " << p.path().string() << '\n';
			result.first = p.path().string().substr(base_length);
			result.second = predict_single(p.path().string(), Roi(), false);
			results->push_back(result);
		}
		else 
		{
			cerr << " Error: " << p.path() << " is not a regular file, skipping ..." << endl;
			_sleep(err_gap / 2);
			continue;
		}
	}
	cout << endl;
	return results;
}

string CellClassifier::repeat_str(const string &str, int times) 
{
	ostringstream builder;
	for (int i = 0; i < times; i++) 
	{
		builder << str;
	}
	return builder.str();
}

void CellClassifier::print_batch_result_to_console(shared_ptr<vector<NamedPred>> results)
{
	/* Build helper strings */
	int url_max_len = 0, field_len = 16;
	for (const auto &iter : *results) 
	{
		auto this_length = iter.first.length();
		if (this_length > url_max_len) 
		{
			url_max_len = this_length;
		}
	}
	url_max_len += 4;
	auto _field_border = "+-" + string(field_len, '-');
	auto _border = "+-" + string(url_max_len, '-') + repeat_str(_field_border, 2) + "+ ";
	int _bar_count = (_border.length() + 2 - _border.length() % 2) / 2;
	auto _half_delimiter = repeat_str("- ", _bar_count / 2);
	auto _full_delimiter = repeat_str("- ", _bar_count);
	/* Format and output */
	cout << _half_delimiter << "Results " << _half_delimiter << endl << endl;
	cout << std::left;
	cout << " " << _border << endl;
	cout << " "
		<< "| " << setw(url_max_len) << "File"
		<< "| " << setw(field_len) << "Prediction"
		<< "| " << setw(field_len) << "Confidence"
		<< "| " << endl;
	cout << " " << _border << endl;
	int stats[3]{ 0, 0, 0 };
	for (const auto &iter : *results) 
	{
		cout << " ";
		cout << "| " << setw(url_max_len) << iter.first;
		int pred = (iter.second)->first;
		cout << "| " << setw(field_len) << class_names[pred];
		cout << "| " << setw(field_len) << (iter.second)->second[pred];
		cout << "| " << endl;
		stats[pred]++;
	}
	cout << " " << _border << endl << endl;
	double total = results->size();
	cout << " " << results->size() << " images in total";
	for (int i = 0; i < 3; i++) 
	{
		cout << ", " << stats[i] << " " << class_names[i] << " (" << 100 * stats[i] / total << "%)";
	}
	cout << "." << endl << endl;
	cout << _full_delimiter << endl << endl;
}

void CellClassifier::save_batch_result_to_csv(shared_ptr<vector<NamedPred>> results, string file_name)
{
	auto file_path = fs::current_path();
	file_path.append("results");
	fs::create_directories(file_path);
	file_path.append(file_name + ".csv");
	cout << endl << "Saving results ..." << endl;
	ofstream fout(file_path);
	fout << "File, Prediction, Fragmented, Hyperfused, WT" << endl;
	for (const auto &iter : *results)
	{
		fout << iter.first << ", " << class_names[(iter.second)->first];
		for (int i = 0; i < 3; i++)
		{
			fout << ", " << (iter.second)->second[i];
		}
		fout << endl;
	}
	fout.close();
	cout << endl << "Results saved to " << file_path.string() << endl;
}

void CellClassifier::run_shell(void)
{
	/* Print hello message */
	cout << "************************************************************************" << endl;
	cout << "*                           CELL CLASSIFIER                            *" << endl;
	cout << "*                             version 0.5                              *" << endl;
	cout << "************************************************************************" << endl << endl;

	/* Main shell logic */
	while (true)
	{
		/* Read input */
		string input_url;
		cout << "Input Path to an Image or a Folder [ q to quit, h for help ]: " << endl;
		do {
			getline(cin, input_url);
		} while (input_url.empty());
		if (input_url == "q")
		{
			break;
		}
		else if (input_url == "h")
		{
			cout << endl
				<< "You can input either" << endl
				<< "a) a path to an image (e.g. C:\\path\\to\\image.tif) to do inference on a single image, or " << endl
				<< "b) a path to a folder (e.g. C:\\path\\to\\folder) to do batch inference on all the images within the folder." << endl
				<< "   (Nested folders are allowed, cause the files are scanned recursively)" << endl
				<< "NOTE that in both cases, ABSOLUTE PATH is required !" << endl
				<< endl;
			continue;
		}
		cout << endl;

		/* Trim off leading and tailing double quotes */
		if (input_url[0] == '"' && input_url[input_url.length() - 1] == '"') 
		{
			input_url = input_url.substr(1, input_url.length() - 2);
		}

		/* Single image inference */
		if (fs::is_regular_file(input_url)) 
		{
			/* Infer */
			auto results = predict_single(input_url, Roi(), true);
			if (results == nullptr) 
			{
				_sleep(err_gap);
				continue;
			}
			/* Format and output */
			auto _half_delimiter = repeat_str("- ", 16);
			auto _full_delimiter = repeat_str("- ", 36);
			cout << _half_delimiter << "Results " << _half_delimiter << endl;
			cout << "Prediction: " << class_names[results->first] << endl;
			cout << "Confidence:" << endl;
			for (int i = 0; i < 3; i++) 
			{
				cout << " - " << class_names[i] << ": " << results->second[i] << endl;
			}
			cout << _full_delimiter << endl << endl;
		}

		/* Batch images inference */
		else if (fs::is_directory(input_url)) 
		{
			/* Infer */
			cout << "Start processing ..." << endl << endl;
			auto results = predict_batch(input_url);
			print_batch_result_to_console(results);
			string choice;
			do {
				cout << "Save results to CSV file ([y]/n) ? ";
				getline(cin, choice);
			} while (choice != "" && choice != "y" && choice != "n");
			if (choice == "n") 
			{
				continue;
			}
			string file_name;
			cout << "Enter file name: ";
			do {
				getline(cin, file_name);
			} while (file_name.empty());
			save_batch_result_to_csv(results, file_name);
		}
		else 
		{
			cerr << "Error: Input URL must be an image file or directory. Please Re-Enter ..." << endl;
			_sleep(err_gap);
			continue;
		}
	}
}
