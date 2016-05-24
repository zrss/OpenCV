#include <iostream>
#include <vector>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

using namespace cv;
using namespace std;

#define OPENCV_DEBUG

static Scalar randomColor(RNG& rng) {  
    int icolor = (unsigned) rng;  
    return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);  
}  

int main(int argc, char const *argv[]) {
	Mat bannerTemplate = imread(argv[1], IMREAD_UNCHANGED);

	if (bannerTemplate.data == NULL) {
		cout << "bannerTemplate not load" << endl;
		return -1;
	}

	#ifdef OPENCV_DEBUG
		cout << "Template cols and rows: " << bannerTemplate.cols << " " << bannerTemplate.rows << endl;
	#endif

	ifstream asinRegion("AsinRegion.txt");
	ifstream textRegion("TextRegion.txt");

	RNG rng(0xFFFFFFFF);

	// Draw Asin Regin in Template
	int cntAsinRegion;
	asinRegion >> cntAsinRegion;

	for (int i = 0; i < cntAsinRegion; ++i) {
		int x, y;
		asinRegion >> x >> y;

		int width, height;
		asinRegion >> width >> height;

		#ifdef OPENCV_DEBUG
			cout << "up left " << x << " " << y << ". down right: " << x + width << " " << y + height << endl;
		#endif

		rectangle(bannerTemplate, Point(x, y), Point(x + width, y + height), randomColor(rng)); 
	}

	// Draw Text Regin in Template
	int cntTextRegin;
	textRegion >> cntTextRegin;

	for (int i = 0; i < cntTextRegin; ++i) {
		int x, y;
		textRegion >> x >> y;

		int width, height;
		textRegion >> width >> height;

		rectangle(bannerTemplate, Point(x, y), Point(x + width, y + height), randomColor(rng)); 
	}

	imshow("bannerTemplate", bannerTemplate);
	waitKey();

	asinRegion.close();
	textRegion.close();

	return 0;
}