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
void start_filter();
void on_trackbar(int, void*);
void callbackButton(int, void*);

const int alpha_max = 125;
const int beta_max = 100;
const int gamma_max = 100;
const int prob1_max = 100;
const int prob2_max = 100;
const int prob3_max = 100;
int alpha;
int beta, gamma;
int prob1, prob2, prob3;
int R, G, B, H, S, V, G2;
Mat src1;
vector<Mat> channels;
vector<vector<ERStat> > regions;
int nobjects;
vector<int> xmin, ymin, xmax, ymax;
int count_image = 0;

int main(int argc, const char * argv[])
{
	cout << "Start.." << endl;
	alpha = 10;
	beta = 1;
	gamma = 2;
	prob1 = 0;
	prob2 = 0;
	prob3 = 0;

	namedWindow("Threshold", WINDOW_AUTOSIZE);
	createTrackbar("Threshold", "Threshold", &alpha, alpha_max, on_trackbar);
	createTrackbar("Min Dist", "Threshold", &beta, beta_max, on_trackbar);
	createTrackbar("Max Dist", "Threshold", &gamma, gamma_max, on_trackbar);
	createTrackbar("Min prob", "Threshold", &prob1, prob1_max, on_trackbar);
	createTrackbar("Max prob", "Threshold", &prob2, prob2_max, on_trackbar);
	createTrackbar("Min prob2", "Threshold", &prob3, prob3_max, on_trackbar);
	
	namedWindow("Channels", WINDOW_AUTOSIZE);
	createTrackbar("ChannelR", "Channels", &R, 1, callbackButton);
	createTrackbar("ChannelG", "Channels", &G, 1, callbackButton);
	createTrackbar("ChannelB", "Channels", &B, 1, callbackButton);
	createTrackbar("ChannelH", "Channels", &H, 1, callbackButton);
	createTrackbar("ChannelS", "Channels", &S, 1, callbackButton);
	createTrackbar("ChannelV", "Channels", &V, 1, callbackButton);
	createTrackbar("ChannelG2", "Channels", &G2, 1, callbackButton);

	//read csv file
	ifstream file("E:\\WORK\\AID\\SVN_xAID\\ClassifierData\\OCR\\TextLocalization\\Dataset\\TelegraTestSet\\LPR\\licenseplate\\LPR_10113431\\2017-06-25\\Annotations\\ispravljeno.csv");
	string   line;

	while (getline(file, line))
	{
	
		separate_line(line, nobjects, xmin, ymin, xmax, ymax);

		src1 = imread(line);

		resize(src1, src1, Size(), 0.3, 0.3, INTER_LINEAR);
		for (int i = 0; i < xmin.size(); i++)
		{
			xmin[i] = xmin[i] * 0.3;
			ymin[i] = ymin[i] * 0.3;
			xmax[i] = xmax[i] * 0.3;
			ymax[i] = ymax[i] * 0.3;
		}

		count_image++;
		callbackButton(R, 0);
		waitKey(0);
		xmin.clear();
		ymin.clear();
		xmax.clear();
		ymax.clear();

	}
	file.close();
}

void on_trackbar(int, void*)
{
	start_filter();
}

void callbackButton(int, void*)
{
	channels.clear();
	vector<Mat> channels_temp;
	if (R == 0 && G == 0 && B == 0 && H == 0 && S == 0 && V == 0 && G2 == 0)
	{
		Mat mask = Mat::zeros(src1.rows, src1.cols, CV_8UC1);
		channels.push_back(mask);
	}
	if (R == 1 || G == 1 || B == 1 || G2 == 1)
	{
		computeNMChannels(src1, channels_temp, 0);
		Mat channelR = channels_temp[0];
		Mat channelG = channels_temp[1];
		Mat channelB = channels_temp[2];
		Mat channelG2 = channels_temp[4];
		if (R==1)
		channels.push_back(channelR);
		if (G == 1)
			channels.push_back(channelG);
		if (B == 1)
			channels.push_back(channelB);
		if (G2 == 1)
			channels.push_back(channelG2);
	}
	if (H == 1 || S == 1 || V == 1)
	{
		computeNMChannels(src1, channels_temp, 1);
		Mat channelH = channels_temp[0];
		Mat channelS = channels_temp[1];
		Mat channelV = channels_temp[2];
		if (H == 1)
			channels.push_back(channelH);
		if (S == 1)
			channels.push_back(channelS);
		if (V == 1)
			channels.push_back(channelV);
	}
	on_trackbar(alpha, 0);
}


void start_filter()
{
	Mat src = src1.clone();
	
	//// Extract channels to be processed individually
	//computeNMChannels(src, channels, 0);

	//Mat channel1 = channels[3];
	////Mat channel2 = channels[4];
	//channels.clear();
	//channels.push_back(channel1);
	////channels.push_back(channel2);

	//int cn = (int)channels.size();
	////Append negative channels to detect ER- (bright regions over dark background)
	//for (int c = 0; c < cn - 1; c++)
	//	channels.push_back(255 - channels[c]);

	float beta_scaled = float(beta) / 100000;
	float gamma_scaled = float(gamma) / 10000;
	float prob1_scaled = float(prob1) / 100.f;
	float prob2_scaled = float(prob2) / 100.f;
	float prob3_scaled = float(prob3) / 100.f;

	cout << "alpha: " << alpha << endl;
	cout << "beta_scaled: " << beta_scaled << endl;
	cout << "gamma_scaled: " << gamma_scaled << endl;
	cout << "prob1_scaled: " << prob1_scaled << endl;
	cout << "prob2_scaled: " << prob2_scaled << endl;
	cout << "prob3_scaled: " << prob3_scaled << endl;

	// Create ERFilter objects with the 1st and 2nd stage default classifiers
	Ptr<ERFilter> er_filter1 = createERFilterNM1(loadDummyClassifier(), alpha, beta_scaled, gamma_scaled, prob1_scaled, false, prob2_scaled);
	Ptr<ERFilter> er_filter2 = createERFilterNM2(loadDummyClassifier(), prob3_scaled);
	vector<vector<ERStat> >regions(channels.size());
	// Apply the default cascade classifier to each independent channel (could be done in parallel)

	for (int c = 0; c < (int)channels.size(); c++)
	 {
		er_filter1->run(channels[c], regions[c]);
		er_filter2->run(channels[c], regions[c]);
	}

	// Detect character groups
	vector< vector<Vec2i> > region_groups;
	vector<Rect> groups_boxes;
	//erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_HORIZ);
	//erGrouping(src, channels, regions, region_groups, groups_boxes, ERGROUPING_ORIENTATION_ANY, "E:\\opencv3_2\\sources\\modules\\text\\samples/trained_classifier_erGrouping.xml", 0.1);

	//crtanje grupa
	groups_draw(src, groups_boxes);
	for (int i = 0; i < nobjects; i++)
	{
		rectangle(src, Point(xmin[i], ymin[i]), Point(xmax[i], ymax[i]), Scalar(0, 0, 255), 2, 8);
	}
	imshow("Detections", src);

	//crtanje regija
	Mat src2 = src.clone();
	for (int c = 0; c < (int)channels.size(); c++)
	{

		for (int i = 0; i < ((regions)._Myfirst)[c].size(); i++)
		{
			rectangle(src2, ((((regions)._Myfirst)[c])._Myfirst)[i].rect, Scalar(255, (c * 125), 0), 3, 8);
			imshow("Threshold", src2);
		}
	}

	cout << "**********************************************************************************" << endl << endl;

	er_filter1.release();
	er_filter2.release();
	if (!groups_boxes.empty())
	{
		groups_boxes.clear();
	}

	er_show(channels, regions);
	
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
		imshow("Binary", dst);
	}
	channels.clear();
	regions.clear();

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
