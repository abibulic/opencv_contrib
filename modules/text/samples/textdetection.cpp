/*
* textdetection.cpp
*
* A demo program of the Extremal Region Filter algorithm described in
* Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
*
* Created on: Sep 23, 2013
*     Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>
*/

#include "opencv2/text.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::text;

void show_help_and_exit(const char *cmd);
void groups_draw(Mat &src, vector<Rect> &groups);
void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions);
void separate_line(string &line, int &nobjects, vector<int> &xmin, vector<int> &ymin, vector<int> &xmax, vector<int> &ymax);
void CountDetection(vector<Rect> groups_boxes, vector<int> xmin, vector<int> ymin, vector<int> xmax, vector<int> ymax, int &good_detect, int &bad_detect, int &false_detect);
void CountRegionDetect(vector<ERStat> region_rect, vector<int> xmin, vector<int> ymin, vector<int> xmax, vector<int> ymax, vector<int> &good_region, vector<int> &bad_region, vector<int> &false_region);

int main(int argc, const char * argv[])
{
	cout << "Start.." << endl;

	//read csv file
	ifstream file("E:\\WORK\\AID\\SVN_xAID\\ClassifierData\\OCR\\TextLocalization\\Dataset\\TelegraTestSet\\LPR\\licenseplate\\LPR_10113431\\2017-06-25\\Annotations\\ispravljeno.csv");
	string   line;

	int nobjects;
	vector<int> xmin, ymin, xmax, ymax;
	int count_image = 0;

	//write csv file
	ofstream myfile;
	myfile.open("E:\\WORK\\AID\\SVN_xAID\\ClassifierData\\OCR\\TextLocalization\\Paths\\TelegraTestSet\\PTZ\\text_precision_check.csv");
	myfile << "ImageNO; true; bad; false\n";
	ofstream myfile2;
	myfile2.open("E:\\WORK\\AID\\SVN_xAID\\ClassifierData\\OCR\\TextLocalization\\Paths\\TelegraTestSet\\PTZ\\text_precision_check2.csv");
	myfile2 << "Channels; true; bad; false\n";

	while (getline(file, line))
	{
		separate_line(line, nobjects, xmin, ymin, xmax, ymax);
		/*
		Mat src = imread("E:\\WORK\\AID\\SVN_xAID\\ClassifierData\\OCR\\TextLocalization\\Dataset\\TelegraTestSet\\LPR\\licenseplate\\LPR_10113431\\2017-06-25\\IMG\\2017-06-25_22-33-23-063_ZG976TF-HRV.jpg");
		xmin[0] = 650;
		ymin[0] = 652;
		xmax[0] = 853;
		ymax[0] = 688;
		*/
		
		Mat src = imread(line);

		resize(src, src, Size(), 0.3, 0.3, INTER_LINEAR);
		for (int i = 0; i < xmin.size(); i++)
		{
			xmin[i] = xmin[i] * 0.3;
			ymin[i] = ymin[i] * 0.3;
			xmax[i] = xmax[i] * 0.3;
			ymax[i] = ymax[i] * 0.3;
		}

		count_image++;

		//probaa
		Mat maskedImage;
		Mat mask(src.size(), src.type());
		mask.setTo(cv::Scalar(0, 0, 0));

		for (size_t i = 0; i < xmin.size(); i++)
		{
			rectangle(mask, Point(xmin[i], ymin[i]), Point(xmax[i], ymax[i]), Scalar(255, 255, 255), -1, 8, 0);
		}

		src.copyTo(src, mask);

		// Extract channels to be processed individually
		vector<Mat> channels;
		computeNMChannels(src, channels, 0);

		//proba
		/*
		Mat hsv;
		cvtColor(src, hsv, COLOR_RGB2HSV);
		vector<Mat> channelsHSV;
		split(src, channelsHSV);
		channels.push_back(channelsHSV[2]);
		*/

		int cn = (int)channels.size();
		//test
		//cn = 1;
		//Append negative channels to detect ER- (bright regions over dark background)
		for (int c = 0; c < cn - 1; c++)
			channels.push_back(255 - channels[c]);
		
		//show channels
		/*
		for (int i = 0; i < channels.size(); i++)
		{
		imshow("1" + to_string(i), channels[i]);
		waitKey(10);
		}
		*/

		// Create ERFilter objects with the 1st and 2nd stage default classifiers
		Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("E:\\WORK\\builds\\build_text_detection\\bin\\Debug\\trained_classifierNM1.xml"), 16, 0.00003f, 0.0005f, 0.2f, true, 0.1f);
		Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("E:\\WORK\\builds\\build_text_detection\\bin\\Debug\\trained_classifierNM2.xml"), 0.4f);

		vector<vector<ERStat> > regions(channels.size());
		// Apply the default cascade classifier to each independent channel (could be done in parallel)
		cout << "Image" << count_image << ": " << "Extracting Class Specific Extremal Regions from " << (int)channels.size() << " channels ..." << endl;
		cout << "    (...) this may take a while (...)" << endl << endl;
		for (int c = 0; c < (int)channels.size(); c++)
		{
			er_filter1->run(channels[c], regions[c]);
			er_filter2->run(channels[c], regions[c]);
		}

		//crtaj regije
		/*
		Mat src2 = src.clone();
		for (int c = 0; c < (int)channels.size(); c++) 
		{

		for (int i = 0; i < ((regions)._Myfirst)[c].size(); i++)
		{
		rectangle(src2, ((((regions)._Myfirst)[c])._Myfirst)[i].rect, Scalar(255, (c * 28), 0), 3, 8);
		imshow("rect", src2);
		waitKey(1);
		}

		}
		*/
		//evaluate region detection
		vector <int> good_region;
		vector <int> bad_region;
		vector <int> false_region;
		for (int c = 0; c < (int)channels.size(); c++)
		{
			CountRegionDetect(regions[c], xmin, ymin, xmax, ymax, good_region, bad_region, false_region);

		}

		// Detect character groups
		cout << "Grouping extracted ERs ... " << endl << endl;
		vector< vector<Vec2i> > region_groups;
		vector<Rect> groups_boxes;
		erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_HORIZ);
		//erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_ANY, "E:\\opencv3_2\\sources\\modules\\text\\samples/trained_classifier_erGrouping.xml", 0.1);

		//evaluate grouped detection
		int good_detect = 0;
		int bad_detect = 0;
		int false_detect = 0;
		CountDetection(groups_boxes, xmin, ymin, xmax, ymax, good_detect, bad_detect, false_detect);

		//parsiranje podataka	

		myfile << "image" << count_image << ";" << good_detect << "; " << bad_detect << "; " << false_detect << "\n";
		myfile2 << "image" << count_image << "\n";
		for (int c = 0; c < (int)channels.size(); c++)
		{
			myfile2 << "channel" << c << ";" << good_region[c] << "; " << bad_region[c] << "; " << false_region[c] << "\n";
		}

		//vizualizacija
		
		groups_draw(src, groups_boxes);
		for (int i = 0; i < nobjects; i++)
		{
			rectangle(src, Point(xmin[i], ymin[i]), Point(xmax[i], ymax[i]), Scalar(0, 0, 255), 2, 8);
		}
		imshow("grouping", src);


		cout << "Done!" << endl << endl;
		cout << "Press 'space' to show the extracted Extremal Regions, any other key to exit." << endl << endl;

		if ((waitKey() & 0xff) == ' ')
		{
			//er_show(channels, regions);
			Mat src2 = src.clone();
			for (int c = 0; c < (int)channels.size(); c++)
			{

				for (int i = 0; i < ((regions)._Myfirst)[c].size(); i++)
				{
					rectangle(src2, ((((regions)._Myfirst)[c])._Myfirst)[i].rect, Scalar(255, (c * 28), 0), 3, 8);
					imshow("rect", src2);
					waitKey(1);
				}
			}
		}
		
		// memory clean-up
		er_filter1.release();
		er_filter2.release();
		regions.clear();
		if (!groups_boxes.empty())
		{
			groups_boxes.clear();
		}

		xmin.clear();
		ymin.clear();
		xmax.clear();
		ymax.clear();

	}
	file.close();
	myfile.close();
	myfile2.close();
}


void groups_draw(Mat &src, vector<Rect> &groups)
{
	for (int i = (int)groups.size() - 1; i >= 0; i--)
	{
		if (src.type() == CV_8UC3)
			rectangle(src, groups.at(i).tl(), groups.at(i).br(), Scalar(0, 255, 255), 3, 8);
		else
			rectangle(src, groups.at(i).tl(), groups.at(i).br(), Scalar(255), 3, 8);
	}
}

void er_show(vector<Mat> &channels, vector<vector<ERStat> > &regions)
{
	for (int c = 0; c < (int)channels.size(); c++)
	{
		Mat dst = Mat::zeros(channels[0].rows + 2, channels[0].cols + 2, CV_8UC1);
		for (int r = 0; r < (int)regions[c].size(); r++)
		{
			ERStat er = regions[c][r];
			if (er.parent != NULL) // deprecate the root region
			{
				int newMaskVal = 255;
				int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
				floodFill(channels[c], dst, Point(er.pixel%channels[c].cols, er.pixel / channels[c].cols),
					Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
			}
		}
		char buff[10]; char *buff_ptr = buff;
		sprintf(buff, "channel %d", c);
		imshow(buff_ptr, dst);
	}
	waitKey(-1);
}

void separate_line(string &line, int &nobjects, vector<int> &xmin, vector<int> &ymin, vector<int> &xmax, vector<int> &ymax)
{

	stringstream linestream(line);
	string data1;
	string data2;

	// read up-to the first ; (discard ;)
	getline(linestream, data1, '\;');
	line.erase(0, data1.size() + 1);

	//get number of objects
	for (int i = 0; i < 6; i++)
	{
		stringstream linestream(line);
		getline(linestream, data1, '\;');
		line.erase(0, data1.size() + 1);
	}
	//nobjects = atoi(data2.c_str());
	nobjects = 1;

	//erase to position
	for (int nobj = 0; nobj < nobjects; nobj++)
	{
		/*
		for (int i = 0; i < 1; i++)
		{
			stringstream linestream(line);
			getline(linestream, data2, '\;');
			line.erase(0, data2.size() + 1);
		}
		*/
		for (int i = 0; i < 4; i++)
		{
			stringstream linestream(line);
			getline(linestream, data2, '\;');
			line.erase(0, data2.size() + 1);
			switch (i)
			{
			case 0:
				xmin.push_back(atoi(data2.c_str()));
				break;
			case 1:
				ymin.push_back(atoi(data2.c_str()));
				break;
			case 2:
				xmax.push_back(atoi(data2.c_str()));
				break;
			case 3:
				ymax.push_back(atoi(data2.c_str()));
				break;
			}
		}
	}
	line = "E:\\WORK\\AID\\SVN_xAID\\ClassifierData\\OCR\\TextLocalization\\Dataset\\TelegraTestSet\\LPR\\licenseplate\\LPR_10113431\\2017-06-25\\IMG\\" + data1;
}

void CountDetection(vector<Rect> groups_boxes, vector<int> xmin, vector<int> ymin, vector<int> xmax, vector<int> ymax, int &good_detect, int &bad_detect, int &false_detect)
{
	for (int i = 0; i < xmin.size(); i++)
	{
		Rect A(xmin[i], ymin[i], (xmax[i] - xmin[i]), (ymax[i] - ymin[i]));

		for (int j = 0; j < groups_boxes.size(); j++)
		{
			int intersects = (A & groups_boxes[j]).area();
			//prihvati 50% greske
			//cout << A << "  " << groups_boxes[j] << "  " << intersects << endl;
			//cout << A.area() << "  " << groups_boxes[j].area() << endl;
			if (intersects >= int(A.area())*0.5 && intersects >= int(groups_boxes[j].area())*0.5)
			{
				good_detect++;
			}
			if ((intersects < int(A.area())*0.5 || intersects < int(groups_boxes[j].area())*0.5) && intersects > 0)
			{
				bad_detect++;
			}
			if (intersects == 0)
			{
				false_detect++;
			}
		}
	}
	/*
	cout << "true: " << good_detect << endl;
	cout << "bad: " << bad_detect << endl;
	cout << "false: " << false_detect << endl;
	*/
}

void CountRegionDetect(vector<ERStat> region_rect, vector<int> xmin, vector<int> ymin, vector<int> xmax, vector<int> ymax, vector<int> &good_region, vector<int> &bad_region, vector<int> &false_region)
{
	int good = 0;
	int bad = 0;
	int fals = 0;
	for (int i = 0; i < xmin.size(); i++)
	{
		Rect A(xmin[i], ymin[i], (xmax[i] - xmin[i]), (ymax[i] - ymin[i]));
		for (int j = 0; j < region_rect.size(); j++)
		{

			int intersects = (A & (((region_rect)._Myfirst)[j]).rect).area();

			if (intersects <= int(A.area()) && intersects <= (((region_rect)._Myfirst)[j]).rect.area() && intersects >(((region_rect)._Myfirst)[j]).rect.area()*0.8)
			{
				good++;
			}
			if (intersects <= (((region_rect)._Myfirst)[j]).rect.area()*0.8 && intersects > 0)
			{
				bad++;
			}
			if (intersects == 0)
			{
				fals++;
			}
		}
	}
	good_region.push_back(good);
	bad_region.push_back(bad);
	false_region.push_back(fals);
	/*
	cout << "true: " << good << endl;
	cout << "bad: " << bad << endl;
	cout << "false: " << fals << endl;
	*/
}
