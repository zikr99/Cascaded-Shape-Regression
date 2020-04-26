#include "PIF.h"

#define it at<int>
#define db at<double>
#define SQR(x) (x)*(x)

float BiIterpolation(IplImage* src_image, float x, float y) //without checking out of image region
{
    if (x < 0) x = 0;
	if (x > (src_image->width - 2)) x = src_image->width - 2;
	if (y < 0) y = 0;
	if (y > (src_image->height - 2)) y = src_image->height - 2;

	int lx = floor(x);
	int upx = lx + 1;
	int ly = floor(y);
	int upy = ly + 1;

	float s = x - lx;
	float t = y - ly;

	return ((1 - s)*(1 - t)*CV_IMAGE_ELEM(src_image, uchar, ly, lx) +
		(1 - s)*t*CV_IMAGE_ELEM(src_image, uchar, upy, lx) +
		s*(1 - t)*CV_IMAGE_ELEM(src_image, uchar, ly, upx) +
		s*t*CV_IMAGE_ELEM(src_image, uchar, upy, upx))/255.0f;
}

double bilinear_interplotation(IplImage* src_image, double x, double y)
{
    if (x < 0) x = 0;
	if (x > (src_image->width - 2)) x = src_image->width - 2;
	if (y < 0) y = 0;
	if (y > (src_image->height - 2)) y = src_image->height - 2;

	int lx = floor(x);
	int upx = lx + 1;
	int ly = floor(y);
	int upy = ly + 1;

	double s = x - lx;
	double t = y - ly;

	return ((1 - s)*(1 - t)*CV_IMAGE_ELEM(src_image, uchar, ly, lx) +
		(1 - s)*t*CV_IMAGE_ELEM(src_image, uchar, upy, lx) +
		s*(1 - t)*CV_IMAGE_ELEM(src_image, uchar, ly, upx) +
		s*t*CV_IMAGE_ELEM(src_image, uchar, upy, upx))/255.0f;
}

void PoseEllipse::calcFeature(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr)
{
    cv::Point2f *impositions = new cv::Point2f [2*nFeatures];

    to_img_positions(positions, nFeatures, impositions);

    for (int i = 0; i < nFeatures; i++)
        ftr[i] = bilinear_interplotation(&(IplImage)(img->featureChannels[positions[i].channel0]),
            impositions[2*i].x, impositions[2*i].y) -
            bilinear_interplotation(&(IplImage)(img->featureChannels[positions[i].channel1]),
            impositions[2*i + 1].x, impositions[2*i + 1].y);

    delete [] impositions;
}

void PoseEllipse::calcOneFeature(const ImageSample* img, const DoublePoint &position, double &ftr)
{
    cv::Point2f *impositions = new cv::Point2f [2];

    vector<DoublePoint> positions;
    positions.push_back(position);

    to_img_positions(positions, 1, impositions);

    ftr = bilinear_interplotation(&(IplImage)(img->featureChannels[position.channel0]),
        impositions[0].x, impositions[0].y) -
        bilinear_interplotation(&(IplImage)(img->featureChannels[position.channel1]),
        impositions[1].x, impositions[1].y);

    delete [] impositions;
}

void PoseEllipse::calcFeatureComp(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr)
{
    cv::Point2f *impositions = new cv::Point2f [2*nFeatures];

    to_img_positions(positions, nFeatures, impositions);

    for (int i = 0; i < nFeatures; i++)
        ftr[i] = bilinear_interplotation(&(IplImage)(img->featureChannels[positions[i].channel0]),
            impositions[2*i].x, impositions[2*i].y);

    delete [] impositions;
}

void drawOnImage(IplImage* image, PoseEllipse pose, cv::Scalar color, int radius, int thickness)
{
    cv::Mat img = cv::Mat(image);

    double scale1 = pow((double)2, pose.scale);
    double c1 = scale1*cos(pose.alpha);
    double s1 = scale1*sin(pose.alpha);
    double ratio1 = pow((double)2.0, pose.ratio);

    double x1, y1, x2, y2;

    x1 = pose.x + c1;
    y1 = pose.y + s1;
    x2 = pose.x - c1;
    y2 = pose.y - s1;
    line(img, cv::Point((int)x1, int(y1)), cv::Point((int)x2, int(y2)), color, radius, thickness);

    x1 = pose.x + ratio1*s1;
    y1 = pose.y - ratio1*c1;
    x2 = pose.x - ratio1*s1;
    y2 = pose.y + ratio1*c1;
    line(img, cv::Point((int)x1, int(y1)), cv::Point((int)x2, int(y2)), color, radius, thickness);
}

