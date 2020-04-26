#ifndef PIF2
#define PIF2

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <math.h>
#include <vector>
#include "PIF.h"
#include "image_sample.hpp"
#include "AAMLibrary/AAM_Shape.h"

#define N_POINTS 19

#define DIST_POINTS_ROT (3.1415926535897932384626433f/12.0f)
#define DIST_POINTS_SCALE 0.06f
#define DIST_POINTS_TRANS 0.1f
#define RANGE_POINTS 64

struct PosePoint
{
	double points[2*N_POINTS];

	PosePoint& operator = (const PosePoint &d)
	{
		for (int i = 0; i < 2*N_POINTS; i++)
            points[i] = d.points[i];

		return *this;
	}

	PosePoint operator + (const PosePoint &p) const
	{
	    PosePoint r;

	    for (int i = 0; i < 2*N_POINTS; i++)
            r.points[i] = points[i] + p.points[i];

		return r;
	}

	PosePoint operator - (const PosePoint &p) const
	{
	    PosePoint r;

	    for (int i = 0; i < 2*N_POINTS; i++)
            r.points[i] = points[i] - p.points[i];

		return r;
	}

	PosePoint operator * (const PosePoint &p) const
	{
	    PosePoint r;

	    for (int i = 0; i < 2*N_POINTS; i++)
            r.points[i] = points[i]*p.points[i];

		return r;
	}

	PosePoint operator / (float p) const
	{
		PosePoint r;

		for (int i = 0; i < 2*N_POINTS; i++)
            r.points[i] = points[i]/p;

		return r;
	}

	void setzero()
	{
	    for (int i = 0; i < 2*N_POINTS; i++)
            points[i] = 0.0f;
	}

	static int numvars()
	{
	    return 2*N_POINTS;
	}

	void Pose2Mat(cv::Mat &mat)
    {
        mat = cv::Mat::zeros(1, 2*N_POINTS, CV_64F);

        for (int i = 0; i < 2*N_POINTS; i++)
            mat.at<double>(0, i) = points[i];
    }

    void Mat2Pose(cv::Mat &mat)
    {
        for (int i = 0; i < 2*N_POINTS; i++)
            points[i] = mat.at<double>(0, i);
    }

    void Pose2Shape(AAM_Shape &shape)
    {
        shape.resize(N_POINTS);

        for (int i = 0; i < N_POINTS; i++)
        {
            shape[i].x = points[2*i];
            shape[i].y = points[2*i + 1];
        }
    }

    void Shape2Pose(AAM_Shape &shape)
    {
        for (int i = 0; i < N_POINTS; i++)
        {
            points[2*i] = shape[i].x;
            points[2*i + 1] = shape[i].y;
        }
    }

	PosePoint composeWithTarget(const PosePoint &p) const;
	PosePoint deltaInTarget(const PosePoint &p) const;

	static int is_equal(const void* _r1, const void* _r2, void*);
	PosePoint disturb();

	static PosePoint calcPose(float* asmshape)
	{
	    int pinds[N_POINTS] = {21, 25, 24, 18, 19, 15, 27, 29, 28, 31, 32, 30,
            35, 57, 39, 44, 56, 50, 7};

	    PosePoint pe;

	    for (int i = 0; i < N_POINTS; i++)
	    {
	        pe.points[2*i] = asmshape[2*pinds[i]];
	        pe.points[2*i + 1] = asmshape[2*pinds[i] + 1];
	    }

        return pe;
	}

    double calcCLoss(PosePoint pe, float* weights, int priority = -1)
    {
        PosePoint dy = deltaInTarget(pe);
        dy = dy*dy;

        cv::Mat matdy;
        dy.Pose2Mat(matdy);

        int nmult = matdy.cols - 1;

        double loss = 0;

        for (int i = 0; i < matdy.cols; i++)
            if (i == priority)
                loss = loss + sqrt(matdy.at<double>(0, i))*weights[i]*nmult;
            else
                loss = loss + sqrt(matdy.at<double>(0, i))*weights[i];

        return loss;
    }

    double calcMLoss(PosePoint pe, float* weights, int priority = -1)
    {
        PosePoint dy = *this - pe;
        dy = dy*dy;

        cv::Mat matdy;
        dy.Pose2Mat(matdy);

        int nmult = matdy.cols - 1;

        double loss = 0;

        for (int i = 0; i < matdy.cols; i++)
            if (i == priority)
                loss = loss + sqrt(matdy.at<double>(0, i))*weights[i]*nmult;
            else
                loss = loss + sqrt(matdy.at<double>(0, i))*weights[i];

        return loss;
    }

	static void generate_random_positions(vector<DoublePoint> &positions, int nFeatures)
	{
        double x0, y0, x1, y1;

        positions.clear();
        int count = 0;

        while (count < nFeatures)
        {
            x0 = (2*RANGE_POINTS)*(double)rand()/RAND_MAX - RANGE_POINTS;
            y0 = (2*RANGE_POINTS)*(double)rand()/RAND_MAX - RANGE_POINTS;

            x1 = (2*RANGE_POINTS)*(double)rand()/RAND_MAX - RANGE_POINTS;
            y1 = (2*RANGE_POINTS)*(double)rand()/RAND_MAX - RANGE_POINTS;

            if ((sqrt(x0*x0 + y0*y0) < RANGE_POINTS) && (sqrt(x1*x1 + y1*y1) < RANGE_POINTS))
            {
                DoublePoint pnt;

                pnt.x0 = x0;
                pnt.y0 = y0;

                pnt.x1 = x1;
                pnt.y1 = y1;

                pnt.pointidx0 = rand()%N_POINTS;
                pnt.pointidx1 = rand()%N_POINTS;

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

            cpos.generate(2*RANGE_POINTS, &rng);
            cpos.extra0 = rand()%N_POINTS;
            cpos.extra1 = rand()%N_POINTS;

            positions.push_back(cpos);
        }
	}

	void to_img_positions(const vector<DoublePoint> positions, int nFeatures, cv::Point2f* impositions);

	void calcFeature(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr);
	void calcOneFeature(const ImageSample* img, const DoublePoint &position, double &ftr);

	void calcFeatureComp(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr);

    void calcFeature(const ImageSample* img, const vector<SimplePixelFeature> &test, double* ftr) const {}
    void calcOneFeature(const ImageSample* img, const SimplePixelFeature &test, double &ftr) const {}

    void Write(ofstream &of)
    {
        for (int i = 0; i < 2*N_POINTS; i++)
            of << (double)points[i] << endl;
    }

    void Read(ifstream &ifs)
    {
        for (int i = 0; i < 2*N_POINTS; i++)
            ifs >> points[i];
    }
};

void drawOnImage(IplImage* image, PosePoint &pose, cv::Scalar color = cv::Scalar(255,255,255), int radius = 3, int thickness = 2);
void drawOnImage(IplImage* image, cv::Point2f* points, int npoints, cv::Scalar color = cv::Scalar(255,255,255), int radius = 3, int thickness = 2);

void CalcSimT(cv::Mat &src, cv::Mat &dst, double &a, double &b, double &tx, double &ty);
void invSimT(double a1, double b1, double tx1, double ty1, double& a2, double& b2, double& tx2, double& ty2);
void SimT(cv::Mat &s, double a, double b, double tx, double ty);

#endif
