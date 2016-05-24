#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

static int flags = 4 + (255 << 8) + FLOODFILL_FIXED_RANGE;

int main(int argc, char** argv) {

    Mat image = imread(argv[1], 1);
    
    if (image.empty()) {
        cout << "Image empty" << endl;
        return 0;
    }

    Mat mask(image.rows + 2, image.cols + 2, CV_8UC1, Scalar::all(0));

    Mat cpyImage;
    image.copyTo(cpyImage);

    // up left corner
    Point seed = Point(0, 0);
    floodFill(cpyImage, mask, seed, Scalar(200, 200, 200), 0, Scalar(20, 20, 20), Scalar(20, 20, 20), flags);

    // reverse
    Mat alpha;
    threshold(mask, alpha, 254, 255, CV_THRESH_BINARY_INV);

    // filter
    medianBlur(alpha, alpha, 3);

    // split to 3 channels BGR
    Mat bgr[3];
    split(image, bgr);

    Mat tmpAlpha = alpha(Range(1, alpha.rows - 1), Range(1, alpha.cols - 1));

    // merge into 4 channels BGRA
    Mat bgra[4] = {bgr[0], bgr[1], bgr[2], tmpAlpha};
    Mat png;
    merge(bgra, 4, png);   

    imwrite(argv[2], png);
    
    return 0;
}
