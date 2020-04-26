#ifndef PIF
#define PIF

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <math.h>
#include <vector>
#include "ASMDLL/ASMDLL.h"
#include "image_sample.hpp"

#define DIST_ELLIPSE_ROT (3.1415926535897932384626433f/18.0f)
#define DIST_ELLIPSE_SCALE 0.06f
#define DIST_ELLIPSE_TRANS 0.1f
#define RANGE_ELLIPSE 1.05f

struct DoublePoint
{
    int channel0;
	double x0;
	double y0;
	int pointidx0;

	int channel1;
	double x1;
	double y1;
	int pointidx1;

	void assignHalf(DoublePoint dp, int h1, int h2)
	{
	    if (h1 == 0)
	    {
            if (h2 == 0)
            {
                channel0 = dp.channel0;
                x0 = dp.x0;
                y0 = dp.y0;
                pointidx0 = dp.pointidx0;
            }
            else
            {
                channel0 = dp.channel1;
                x0 = dp.x1;
                y0 = dp.y1;
                pointidx0 = dp.pointidx1;
            }
        }
	    else
	    {
            if (h2 == 0)
            {
                channel1 = dp.channel0;
                x1 = dp.x0;
                y1 = dp.y0;
                pointidx1 = dp.pointidx0;
            }
            else
            {
                channel1 = dp.channel1;
                x1 = dp.x1;
                y1 = dp.y1;
                pointidx1 = dp.pointidx1;
            }
	    }
	}

	void Write(ofstream &of)
    {
        of << (int)channel0 << endl;
        of << (double)x0 << endl;
        of << (double)y0 << endl;
        of << (int)pointidx0 << endl;

        of << (int)channel1 << endl;
        of << (double)x1 << endl;
        of << (double)y1 << endl;
        of << (int)pointidx1 << endl;
    }

    void Read(ifstream &ifs)
    {
        ifs >> channel0;
        ifs >> x0;
        ifs >> y0;
        ifs >> pointidx0;

        ifs >> channel1;
        ifs >> x1;
        ifs >> y1;
        ifs >> pointidx1;
    }
};

struct PoseEllipse
{
	double x;
	double y;
	double alpha;
	double scale;
	double ratio;

	PoseEllipse& operator = (const PoseEllipse &d)
	{
		x = d.x;
		y = d.y;
		alpha = d.alpha;
		scale = d.scale;
		ratio = d.ratio;

		return *this;
	}

	PoseEllipse operator + (const PoseEllipse &p) const
	{
	    PoseEllipse r;

		r.x = x + p.x;
		r.y = y + p.y;
		r.alpha = alpha + p.alpha;
		r.scale = scale + p.scale;
		r.ratio = ratio + p.ratio;

		return r;
	}

	PoseEllipse operator - (const PoseEllipse &p) const
	{
	    PoseEllipse r;

		r.x = x - p.x;
		r.y = y - p.y;
		r.alpha = alpha - p.alpha;
		r.scale = scale - p.scale;
		r.ratio = ratio - p.ratio;

		return r;
	}

	PoseEllipse operator * (const PoseEllipse &p) const
	{
	    PoseEllipse r;

		r.x = x*p.x;
		r.y = y*p.y;
		r.alpha = alpha*p.alpha;
		r.scale = scale*p.scale;
		r.ratio = ratio*p.ratio;

		return r;
	}

	PoseEllipse operator / (float p) const
	{
		PoseEllipse r;

		r.x = x/p;
		r.y = y/p;
		r.alpha = alpha/p;
		r.scale = scale/p;
		r.ratio = ratio/p;

		return r;
	}

	void setzero()
	{
	    x = 0.0f;
	    y = 0.0f;
	    alpha = 0.0f;
	    scale = 0.0f;
	    ratio = 0.0f;
	}

	static int numvars()
	{
	    return 5;
	}

	void Pose2Mat(cv::Mat &mat)
    {
        mat = cv::Mat::zeros(1, 5, CV_64F);

        mat.at<double>(0, 0) = x;
        mat.at<double>(0, 1) = y;
        mat.at<double>(0, 2) = alpha;
        mat.at<double>(0, 3) = scale;
        mat.at<double>(0, 4) = ratio;
    }

    void Mat2Pose(cv::Mat &mat)
    {
        x = mat.at<double>(0, 0);
        y = mat.at<double>(0, 1);
        alpha = mat.at<double>(0, 2);
        scale = mat.at<double>(0, 3);
        ratio = mat.at<double>(0, 4);
    }

	PoseEllipse composeWithTarget(const PoseEllipse &p) const
	{
		PoseEllipse r;

		double scale1 = pow((double)2, scale);
        double c1 = scale1*cos(alpha);
        double s1 = scale1*sin(alpha);

        r.x = x + c1*p.x - s1*p.y;
		r.y = y + s1*p.x + c1*p.y;

		r.scale = scale + p.scale;

        r.alpha = alpha + p.alpha;
        if (r.alpha > 2*CV_PI) r.alpha = r.alpha - floor(r.alpha/2/CV_PI)*2*CV_PI;
        if (r.alpha > CV_PI) r.alpha = r.alpha - 2*CV_PI;

	    r.ratio = ratio + p.ratio;

        return r;
	}

	PoseEllipse deltaInTarget(const PoseEllipse &p) const
	{
		PoseEllipse r;

        double scale1 = pow((double)2, scale);
        double c1 = scale1*cos(alpha);
        double s1 = scale1*sin(alpha);

        double dx = p.x - x;
        double dy = p.y - y;
        double det = c1*c1 + s1*s1;

        r.x = (c1*dx + s1*dy)/det;
		r.y = (-s1*dx + c1*dy)/det;

		r.scale = p.scale - scale;

        r.alpha = p.alpha - alpha;
        if (r.alpha < 0.0f) r.alpha = r.alpha + ceil(-r.alpha/2/CV_PI)*2*CV_PI;
        if (r.alpha > CV_PI) r.alpha = r.alpha - 2*CV_PI;

        r.ratio = p.ratio - ratio;

		return r;
	}

	static int is_equal(const void* _r1, const void* _r2, void*)
    {
        const PoseEllipse* r1 = (const PoseEllipse*)_r1;
        const PoseEllipse* r2 = (const PoseEllipse*)_r2;

        float distance = 0.4f*(pow(2.0f, r1->scale) + pow(2.0f, r2->scale))/2;

        float alpha_distance = CV_PI/18.0f;
        float scale_distance = 0.15f;

        return (abs(r1->x - r2->x) < distance) &&
            (abs(r1->y - r2->y) < distance) &&
            (abs(r1->alpha - r2->alpha) < alpha_distance) &&
            (abs(r1->scale - r2->scale) < scale_distance) &&
            (abs(r1->ratio - r2->ratio) < scale_distance);
    }

	PoseEllipse disturb()
	{
	    PoseEllipse pe;

        double rd = (2*DIST_ELLIPSE_ROT)*(double)rand()/RAND_MAX - DIST_ELLIPSE_ROT;
        pe.alpha = alpha + rd;

        if (pe.alpha > 2*CV_PI) pe.alpha = pe.alpha - floor(pe.alpha/2/CV_PI)*2*CV_PI;
        else if (pe.alpha < 0.0f) pe.alpha = pe.alpha + ceil(-pe.alpha/2/CV_PI)*2*CV_PI;
        if (pe.alpha > CV_PI) pe.alpha = pe.alpha - 2*CV_PI;

        double scale1 = pow((double)2.0, scale);
        rd = (2*DIST_ELLIPSE_SCALE)*(double)rand()/RAND_MAX - DIST_ELLIPSE_SCALE;
        scale1 = scale1*(1.0f + rd);

        pe.scale = log(scale1)/log((double)2.0);
        pe.ratio = ratio;

        double c1 = scale1*cos(pe.alpha);
        double s1 = scale1*sin(pe.alpha);

        double dx = (2*DIST_ELLIPSE_TRANS)*(double)rand()/RAND_MAX - DIST_ELLIPSE_TRANS;
        double dy = (2*DIST_ELLIPSE_TRANS)*(double)rand()/RAND_MAX - DIST_ELLIPSE_TRANS;

        double ratio1 = pow((double)2.0, ratio);
        dx = dx*ratio1;

        pe.x = x + dx*c1 - dy*s1;
        pe.y = y + dx*s1 + dy*c1;

        return pe;
	}

	static PoseEllipse calcPose(float* asmshape)
	{
	    int pinds[3] = {57, 7, 1};

	    PoseEllipse pe;

	    pe.x = asmshape[2*pinds[0]];
	    pe.y = asmshape[2*pinds[0] + 1];

	    double dx = asmshape[2*pinds[1]] - pe.x;
	    double dy = asmshape[2*pinds[1] + 1] - pe.y;

	    pe.alpha = atan2(dy, dx);

	    pe.scale = sqrt(dx*dx + dy*dy);

	    dx = asmshape[2*pinds[2]] - pe.x;
	    dy = asmshape[2*pinds[2] + 1] - pe.y;

	    pe.ratio = sqrt(dx*dx + dy*dy)/pe.scale;

	    pe.scale = log(pe.scale)/log((double)2.0);
	    pe.ratio = log(pe.ratio)/log((double)2.0);

        return pe;
	}

	static PoseEllipse calcPose(float x, float y, float ra, float rb, float theta)
	{
	    PoseEllipse pe;

	    pe.x = x;
	    pe.y = y;

	    pe.scale = ra;

	    pe.ratio = rb/pe.scale;
	    pe.alpha = theta;

	    if (pe.alpha > 2*CV_PI) pe.alpha = pe.alpha - floor(pe.alpha/2/CV_PI)*2*CV_PI;
        else if (pe.alpha < 0.0f) pe.alpha = pe.alpha + ceil(-pe.alpha/2/CV_PI)*2*CV_PI;
        if (pe.alpha > CV_PI) pe.alpha = pe.alpha - 2*CV_PI;

	    pe.scale = log(pe.scale)/log((double)2.0);
	    pe.ratio = log(pe.ratio)/log((double)2.0);

        return pe;
	}

	float* calcASMShape()
	{
	    double scale1 = pow((double)2.0, scale);
	    double ratio1 = pow((double)2.0, ratio);

        CvRect rect;

        rect.x = -(int)round(scale1*ratio1*1.15f);
        rect.y = -(int)round(scale1*0.60f);
        rect.width = (int)round(scale1*ratio1*1.15f) + (int)round(scale1*ratio1*1.15f);
        rect.height = (int)round(scale1*1.05f) + (int)round(scale1*0.60f);

        double theta = alpha - CV_PI/2.0f;
        double cos1 = cos(theta);
        double sin1 = sin(theta);

        int numpoints;
        float *asmshape;

        cv::Mat dump = cv::Mat::zeros(100, 100, CV_8UC3);
        ASMDLLFit(0, numpoints, &asmshape, &(IplImage)dump, &rect, true);

        float *asmpoints = new float [2*numpoints];

        for (int i = 0; i < numpoints; i++)
        {
            float px = asmshape[2*i];
            float py = asmshape[2*i + 1];

            asmpoints[2*i] = cos1*px - sin1*py + x;
            asmpoints[2*i + 1] = sin1*px + cos1*py + y;
        }

        FreeFitResult(&asmshape);

        return asmpoints;
	}

    double calcCLoss(PoseEllipse pe, float* weights, int priority = -1)
    {
        PoseEllipse dy = deltaInTarget(pe);
        dy = dy*dy;

        cv::Mat matdy;
        dy.Pose2Mat(matdy);

        int nmult = matdy.cols - 1;

        double loss = 0;

        for (int i = 0; i < matdy.cols; i++)
            if (i == priority)
                loss = loss + matdy.at<double>(0, i)*weights[i]*nmult;
            else
                loss = loss + matdy.at<double>(0, i)*weights[i];

        return loss;
    }

    double calcMLoss(PoseEllipse pe, float* weights, int priority = -1)
    {
        PoseEllipse dy = *this - pe;
        dy = dy*dy;

        cv::Mat matdy;
        dy.Pose2Mat(matdy);

        int nmult = matdy.cols - 1;

        double loss = 0;

        for (int i = 0; i < matdy.cols; i++)
            if (i == priority)
                loss = loss + matdy.at<double>(0, i)*weights[i]*nmult;
            else
                loss = loss + matdy.at<double>(0, i)*weights[i];

        return loss;
    }

	static void generate_random_positions(vector<DoublePoint> &positions, int nFeatures)
	{
        double x0, y0, x1, y1;

        positions.clear();
        int count = 0;

        while (count < nFeatures)
        {
            x0 = (2*RANGE_ELLIPSE)*(double)rand()/RAND_MAX - RANGE_ELLIPSE;
            y0 = (2*RANGE_ELLIPSE)*(double)rand()/RAND_MAX - RANGE_ELLIPSE;

            x1 = (2*RANGE_ELLIPSE)*(double)rand()/RAND_MAX - RANGE_ELLIPSE;
            y1 = (2*RANGE_ELLIPSE)*(double)rand()/RAND_MAX - RANGE_ELLIPSE;

            if ((sqrt(x0*x0 + y0*y0) < RANGE_ELLIPSE) && (sqrt(x1*x1 + y1*y1) < RANGE_ELLIPSE))
            {
                DoublePoint pnt;

                pnt.x0 = x0;
                pnt.y0 = y0;

                pnt.x1 = x1;
                pnt.y1 = y1;

                pnt.pointidx0 = 0;
                pnt.pointidx1 = 0;

                pnt.channel0 = 0;
                pnt.channel1 = 0;

                positions.push_back(pnt);

                count++;
            }
        }
	}

    template<class FeatureType>
	static void generate_random_positions(vector<FeatureType> positions, int nFeatures)
	{
	    boost::mt19937 rng;
        rng.seed(1);

        positions.clear();

        for (int count = 0; count < nFeatures; count++)
        {
            FeatureType cpos;

            cpos.generate(2*RANGE_ELLIPSE, &rng);
            cpos.extra0 = 0;
            cpos.extra1 = 0;

            positions.push_back(cpos);
        }
	}

	void to_img_positions(const vector<DoublePoint> positions, int nFeatures, cv::Point2f* impositions)
	{
        double scale1 = pow((double)2, scale);
        double c1 = scale1*cos(alpha);
        double s1 = scale1*sin(alpha);
        double ratio1 = pow((double)2.0, ratio);

        double px, py;

        for (int i = 0; i < nFeatures; i++)
        {
            px = positions[i].x0*ratio1;
            py = positions[i].y0;

            impositions[2*i].x = x + c1*px - s1*py;
            impositions[2*i].y = y + s1*px + c1*py;

            px = positions[i].x1*ratio1;
            py = positions[i].y1;

            impositions[2*i + 1].x = x + c1*px - s1*py;
            impositions[2*i + 1].y = y + s1*px + c1*py;
        }
	}

	void calcFeature(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr);
	void calcOneFeature(const ImageSample* img, const DoublePoint &position, double &ftr);

	void calcFeatureComp(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr);

    void calcFeature(const ImageSample* img, const vector<SimplePixelFeature> &test, double* ftr) const
    {
        double scale1 = pow((double)2, scale);
        double c1 = scale1*cos(alpha);
        double s1 = scale1*sin(alpha);
        double ratio1 = pow((double)2.0, ratio);

        double px, py;
        int x1, y1, x2, y2;

        for (int i = 0; i < test.size(); i++)
        {
            px = test[i].point_a.x*ratio1;
            py = test[i].point_a.y;

            x1 = cvRound(x + c1*px - s1*py);
            y1 = cvRound(y + s1*px + c1*py);

            px = test[i].point_b.x*ratio1;
            py = test[i].point_b.y;

            x2 = cvRound(x + c1*px - s1*py);
            y2 = cvRound(y + s1*px + c1*py);

            ftr[i] = img->featureChannels[test[i].featureChannel].at<unsigned char>(y1, x1)
                - img->featureChannels[test[i].featureChannel].at<unsigned char>(y2, x2);
        }
    }

    void calcOneFeature(const ImageSample* img, const SimplePixelFeature &test, double &ftr) const
    {
        double scale1 = pow((double)2, scale);
        double c1 = scale1*cos(alpha);
        double s1 = scale1*sin(alpha);
        double ratio1 = pow((double)2.0, ratio);

        double px, py;
        int x1, y1, x2, y2;

        px = test.point_a.x*ratio1;
        py = test.point_a.y;

        x1 = cvRound(x + c1*px - s1*py);
        y1 = cvRound(y + s1*px + c1*py);

        px = test.point_b.x*ratio1;
        py = test.point_b.y;

        x2 = cvRound(x + c1*px - s1*py);
        y2 = cvRound(y + s1*px + c1*py);

        ftr = img->featureChannels[test.featureChannel].at<unsigned char>(y1, x1)
            - img->featureChannels[test.featureChannel].at<unsigned char>(y2, x2);
    }

    void Write(ofstream &of)
    {
        of << (double)x << endl;
        of << (double)y << endl;
        of << (double)alpha << endl;
        of << (double)scale << endl;
        of << (double)ratio << endl;
    }

    void Read(ifstream &ifs)
    {
        ifs >> x;
        ifs >> y;
        ifs >> alpha;
        ifs >> scale;
        ifs >> ratio;
    }
};

struct PoseEllipseLeaf
{
    int test;
};

double bilinear_interplotation(IplImage* src_image, double x, double y);
void drawOnImage(IplImage* image, PoseEllipse pose, cv::Scalar color = cv::Scalar(255, 255, 255), int radius = 3, int thickness = 2);

#endif
