#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "Median_Blur.h"
using std::cin;
using std::cout;
using std::endl;
using namespace cv;

vector< vector<unsigned> > medianBlurring(vector< vector<unsigned> > img, int radius, string padding_way) {
	Median_Blur MB;
	vector< vector<unsigned> > img_blur = MB.get_median_image(img, radius, padding_way);
	return img_blur;
}

vector< vector<unsigned> > cvtMat2Vector(Mat image) {
	int rows = image.rows;
	int cols = image.cols;
	vector<vector<unsigned> > img(rows, vector<unsigned>(cols));
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			uchar val = image.at<uchar>(i, j);
			img[i][j] = static_cast<unsigned>(val);
		}
	}
	return img;
}

int main() {
	int radius = 2;
	string padding = "REPLICA";
	Mat image = imread("../20190712182540.jpg");
	
	cout << image.size << endl;
	int H = image.rows / 5;
	int W = image.cols / 5;	
	int C = image.channels();
	Size dsize(W, H);
	resize(image, image, dsize);
	cout << H << " " << W << " " << C << endl;
	
	int image_type = image.type();
	cout << image_type << endl;
	
	vector<Mat> BGR;
	split(image, BGR);
	cout << BGR[0].type() << endl; // 0
	vector<vector<unsigned> > B = cvtMat2Vector(BGR[0]);
	vector<vector<unsigned> > G = cvtMat2Vector(BGR[1]);
	vector<vector<unsigned> > R = cvtMat2Vector(BGR[2]);

	vector<vector<unsigned> > b = medianBlurring(B, radius, padding);
	vector<vector<unsigned> > g = medianBlurring(G, radius, padding);
	vector<vector<unsigned> > r = medianBlurring(R, radius, padding);
	
	/* vector<Mat> img_blur_BGR;
	 img_blur_BGR.push_back(Mat(b));
	 img_blur_BGR.push_back(Mat(g, 4));
	 img_blur_BGR.push_back(Mat(r, 4));
	 Mat image_blur;
	 merge(img_blur_BGR, image_blur);
	 imwrite("C:/Users/Thinkpad X1 Carbon/source/repos/medianBlur/medianBlur/medianBlur.jpg", image_blur);*/
	
	/* TODO: vector<vector<unsigned> >转 Mat报错 | C2039 “type”:不是“cv::DataType<T>”的成员
	Mat image_blur;
	if (image.channels() == 1) {
		vector< vector<unsigned> > img = cvtMat2Vector(image);
		vector< vector<unsigned> > img_blur = medianBlurring(img, radius, padding);
		image_blur = Mat(img_blur);
	}
	else if (image.channels() == 3) {
		vector<Mat> BGR;
		split(image, BGR);
		vector<vector<unsigned> > B = cvtMat2Vector(BGR[0]);
		vector<vector<unsigned> > G = cvtMat2Vector(BGR[1]);
		vector<vector<unsigned> > R = cvtMat2Vector(BGR[2]);

		vector<vector<unsigned> > b = medianBlurring(B, radius, padding);
		vector<vector<unsigned> > g = medianBlurring(G, radius, padding);
		vector<vector<unsigned> > r = medianBlurring(R, radius, padding);

		vector<Mat> img_blur_BGR;
		img_blur_BGR.push_back(Mat(b));
		img_blur_BGR.push_back(Mat(g));
		img_blur_BGR.push_back(Mat(r));
		merge(img_blur_BGR, image_blur);
	}
	imwrite("../medianBlur.jpg", image_blur);
	*/
	
	system("pause");
	return 0;
}