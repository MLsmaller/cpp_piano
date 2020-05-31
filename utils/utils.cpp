#include<map>
#include<string>
#include<iostream>
#include<vector>

using namespace std;


map<string, vector<double>> Config(map<string, string>& string_config) {
	string_config["img_paths"]= "E:\\Learning\\visual_studio\\piano\\piano\\video\\video1\\";

	double NUM_CLASSES = 2;
	double KEYBOARD_PALETTE[] = { 0, 0, 0, 64, 0, 128 };

	vector<double> num_classes, keyboard_plaette;
	num_classes.push_back(NUM_CLASSES);
	for (int i = 0; i < (sizeof(KEYBOARD_PALETTE) / sizeof(KEYBOARD_PALETTE[0])); i++) {
		keyboard_plaette.push_back(KEYBOARD_PALETTE[i]);
	}

	map<string, vector<double>>number_config;
	number_config["NUM_CLASSES"] = num_classes;
	number_config["KEYBOARD_PALETTE"] = keyboard_plaette;

	return number_config;
}