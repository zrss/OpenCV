#include <iostream>
#include <vector>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

using namespace cv;
using namespace std;

#define OPENCV_DEBUG_NO

static int flags = 4 + (255 << 8) + FLOODFILL_FIXED_RANGE;

// 逼近矩形
void approachOfRect(Mat& src, Mat& gray, Mat& output, int thresh) {

    int miny = 0, maxy = gray.rows, minx = 0, maxx = gray.cols;
    int colNums = gray.cols * gray.channels();

    bool white = true;

    // up
    for (int i = 0; white && i < gray.rows; ++i) {
        uchar* data = gray.ptr<uchar>(i);
        for (int j = 0; white && j < colNums; ++j) {
            if (data[j] < thresh) {
                white = false;
            }
        }
        if (white) {
            ++miny;
        }
    }

    // down
    white = true;
    for (int i = gray.rows - 1; white && i >= 0; --i) {
        uchar* data = gray.ptr<uchar>(i);
        for (int j = 0; white && j < colNums; ++j) {
            if (data[j] < thresh) {
                white = false;
            }
        }
        if (white) {
            --maxy;
        }
    }

    // left
    white = true;
    for (int j = 0; white && j < colNums; ++j) {
        for (int i = 0; white && i < gray.rows; ++i) {
            uchar* data = gray.ptr<uchar>(i);
            if (data[j] < thresh) {
                white = false;
            }
        }
        if (white) {
            ++minx;
        }
    }

    // right
    white = true;
    for (int j = colNums - 1; white && j >= 0; --j) {
        for (int i = 0; white && i < gray.rows; ++i) {
            uchar* data = gray.ptr<uchar>(i);
            if (data[j] < thresh) {
                white = false;
            }
        }
        if (white) {
            --maxx;
        }
    }

	#ifdef OPENCV_DEBUG
		cout << "rows and cols: " << src.rows << " " << src.cols << endl;

		cout << "Range(" << miny << "," << maxy << ")" << " "
			<< "Range(" << minx << "," << maxx << ")" << endl;
    #endif

    src(Range(miny, maxy), Range(minx, maxx)).copyTo(output);

	#ifdef OPENCV_DEBUG
		cout << "approachOfRect ending" << endl;
	#endif

}

// 将白色背景设置为透明
// 使用FloodFill算法
void AutoFloodFill(Mat& image, Mat& png) {
	Mat mask(image.rows + 2, image.cols + 2, CV_8UC1, Scalar::all(0));

    Mat cpyImage;
    image.copyTo(cpyImage);

    // 4 corner points
    Point seedUpLeft = Point(0, 0);
    Point seedUpRight = Point(image.cols - 1, 0);
    Point seedDownLeft = Point(0, image.rows - 1);
    Point seedDownRight = Point(image.cols - 1, image.rows - 1);
    
    floodFill(cpyImage, mask, seedUpLeft, Scalar(200, 200, 200), 0, Scalar(20, 20, 20), Scalar(20, 20, 20), flags);
    floodFill(cpyImage, mask, seedUpRight, Scalar(200, 200, 200), 0, Scalar(20, 20, 20), Scalar(20, 20, 20), flags);
    floodFill(cpyImage, mask, seedDownLeft, Scalar(200, 200, 200), 0, Scalar(20, 20, 20), Scalar(20, 20, 20), flags);
    floodFill(cpyImage, mask, seedDownRight, Scalar(200, 200, 200), 0, Scalar(20, 20, 20), Scalar(20, 20, 20), flags);
  	
    // reverse
    Mat alpha;
    threshold(mask, alpha, 254, 255, CV_THRESH_BINARY_INV);

    // filter
    // medianBlur(alpha, alpha, 3);

    // split to 3 channels BGR
    Mat bgr[3];
    split(image, bgr);

    Mat tmpAlpha = alpha(Range(1, alpha.rows - 1), Range(1, alpha.cols - 1));

    // merge into 4 channels BGRA
    Mat bgra[4] = {bgr[0], bgr[1], bgr[2], tmpAlpha};
    merge(bgra, 4, png);
}


// 等比例缩放
void uniformScale(Mat& src, Mat& dst, Mat& output) {
	// Resize asin to fit
	bool flagResize = false;
	double scale;

    // 锁定长宽比缩放
    // 高度缩放
    if (src.rows > dst.rows) {
        scale = (double)dst.rows / src.rows;
        flagResize = true;
    }

    // 宽度缩放
    if (src.cols > dst.cols) {
    	if (!flagResize || (int)(scale * src.cols) > dst.cols) {
	        scale = (double)dst.cols / src.cols;
    	}
    	flagResize = true;
    }

    #ifdef OPENCV_DEBUG
        cout << "scale: " << scale << ". " 
        	<< "cols: " << (int)(scale * src.cols) << ". " << "rows: " 
        	<< (int)(scale * src.rows) << "." << endl;
    #endif

    if (flagResize) {
	    Mat afterResize;
    	resize(src, afterResize, Size(), scale, scale, CV_INTER_AREA);
    	output = afterResize;

	    #ifdef OPENCV_DEBUG
        	cout << "afterResize cols and rows: " << afterResize.cols << " " << afterResize.rows << endl;
    	#endif

    	return;
    }

    output = src;

	#ifdef OPENCV_DEBUG
		cout << "uniformScale ending" << endl;
	#endif

}

// png图像融合
void mergePngToJpg(Mat& png, Mat& jpg) {
	for (int i = 0; i < png.rows; ++i) {
		for (int j = 0; j < png.cols; ++j) {
			if (png.at<Vec4b>(i, j)[3] == 255) {
				jpg.at<Vec3b>(i, j)[0] = png.at<Vec4b>(i, j)[0];
				jpg.at<Vec3b>(i, j)[1] = png.at<Vec4b>(i, j)[1];
				jpg.at<Vec3b>(i, j)[2] = png.at<Vec4b>(i, j)[2];				
			}
		}
	}
}

// 生成随机颜色
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

	RNG rng(0xFFFFFFF0);

	ifstream asinRegion("../Resource/AsinRegion.txt");
	ifstream textRegion("../Resource/TextRegion.txt");
	ifstream asinName("../Resource/AsinName.txt");

	// Draw Asin Regin in Template
	int cntAsinRegion;
	asinRegion >> cntAsinRegion;

	for (int i = 0; i < cntAsinRegion; ++i) {
		int x, y;
		asinRegion >> x >> y;

		int width, height;
		asinRegion >> width >> height;

		// TODO: get asin list

		string asinNameText;
		asinName >> asinNameText;

		asinNameText = "../Resource/" + asinNameText;

		#ifdef OPENCV_DEBUG
			cout << asinNameText << endl;
		#endif

		Mat asin = imread(asinNameText, IMREAD_UNCHANGED);

		// TODO: height / width rate

		Mat gray;
		cvtColor(asin, gray, COLOR_RGB2GRAY);

		Mat shrink;
		approachOfRect(asin, gray, shrink, 250);

		Mat regionMat(width, height, CV_8UC1);
		Mat fit;
		uniformScale(shrink, regionMat, fit);

		Mat png = fit;
		AutoFloodFill(fit, png);

		int gapx = (width - png.cols) >> 1;
		int gapy = (height - png.rows) >> 1;

		#ifdef OPENCV_DEBUG
			cout << "original x and y: " << x << ", " << y << endl;
			cout << "offset gapx and gapy: " << gapx << ", " << gapy << endl;	
		#endif

		Mat roiOfAsinRegion = bannerTemplate(Rect(x + gapx, y + gapy, png.cols, png.rows));

		#ifdef OPENCV_DEBUG
		rectangle(bannerTemplate, Point(x, y), Point(x + width, y + height), randomColor(rng)); 
		#endif

		#ifdef OPENCV_DEBUG
		imshow("roiOfAsinRegion", roiOfAsinRegion);
		imshow("png", png);
		#endif

		mergePngToJpg(png, roiOfAsinRegion);
	}

	// Draw Text Regin in Template
	int cntTextRegin;
	textRegion >> cntTextRegin;

	for (int i = 0; i < cntTextRegin; ++i) {
		int x, y;
		textRegion >> x >> y;

		int width, height;
		textRegion >> width >> height;

		// TODO: get text list

		// #ifdef OPENCV_DEBUG
		rectangle(bannerTemplate, Point(x, y), Point(x + width, y + height), randomColor(rng)); 
		// #endif
	}

	imshow("bannerTemplate", bannerTemplate);
	waitKey();

	asinRegion.close();
	textRegion.close();
	asinName.close();

	return 0;
}