#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

using namespace cv;
using namespace std;

#define OPENCV_DEBUG

void approachOfRect(Mat& src, Mat& gray, Mat& output, int thresh) {
    // 逼近矩形
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
    cout << "left useless col: " << minx << endl;
    cout << "right useless col start: " << maxx << endl;
    cout << "up useless col: " << miny << endl;
    cout << "down useless col start: " << maxy << endl; 
    cout << "gray cols and rows: " <<  gray.cols << " " << gray.rows << endl;
#endif

    src(Range(miny, maxy + 1), Range(minx, maxx + 1)).copyTo(output);

#ifdef OPENCV_DEBUG
    imshow("src", src);
    imshow("shrink", output);
    cout << "shring size cols and rows: " << output.cols << " " << output.rows << endl;
#endif

}

int main(int argc, char const *argv[]) {
    Mat src = imread(argv[1], IMREAD_UNCHANGED);

    // 转换成灰度图
    Mat gray;
    cvtColor(src, gray, COLOR_RGB2GRAY);

    Mat shrink;
    approachOfRect(src, gray, shrink, 250);

    Mat dst = imread(argv[2], IMREAD_UNCHANGED);

    if (shrink.rows > dst.rows || shrink.cols > dst.cols) {

    #ifdef OPENCV_DEBUG
        cout << "src is large than background. need to resize." << endl;
    #endif

        double scale;

        // 锁定长宽比缩放
        // 高度缩放
        if (shrink.rows > dst.rows) {
            scale = (double)dst.rows / shrink.rows;
        }

        // 宽度缩放
        if (scale * shrink.cols > dst.cols) {
            scale = (double)dst.cols / shrink.cols;
        }

    #ifdef OPENCV_DEBUG
        cout << "scale: " << scale << ". " << "cols: " << scale * shrink.cols << ". " 
            << "rows: " << scale * shrink.rows << "." << endl;
    #endif

        Mat afterResize;
        resize(shrink, afterResize, Size(), scale, scale, CV_INTER_AREA);

    #ifdef OPENCV_DEBUG
        cout << "afterResize cols and rows: " << afterResize.cols << " " << afterResize.rows << endl;
    #endif

        shrink = afterResize;
    }

    // Create an all white mask
    Mat src_mask = 255 * Mat::ones(shrink.rows, shrink.cols, shrink.depth());

    // The location of the center of the src in the dst
    Point center(dst.cols >> 1, dst.rows >> 1);

    // Seamlessly clone src into dst and put the results in output
    Mat normal_clone;
    Mat mixed_clone;

    seamlessClone(shrink, dst, src_mask, center, normal_clone, NORMAL_CLONE);
    seamlessClone(shrink, dst, src_mask, center, mixed_clone, MIXED_CLONE);

    // Save results
    imshow("normal clone", normal_clone);
    imshow("mixed clone", mixed_clone);
    waitKey();

    return 0;
}