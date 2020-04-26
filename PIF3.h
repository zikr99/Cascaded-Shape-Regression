#ifndef PIF3
#define PIF3

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <math.h>
#include <vector>
#include "PIF2.h"
#include "image_sample.hpp"
#include "AAMLibrary/AAM_Shape.h"

#define N_POINTS_3 6

#define DIST_POINTS_ROT_3 (3.1415926535897932384626433f/24.0f)
#define DIST_POINTS_SCALE_3 0.05f
#define DIST_POINTS_TRANS_3 0.08f
#define RANGE_POINTS 64

struct PoseMViewPoint
{
    double yaw;
	double points[2*N_POINTS_3];

	PoseMViewPoint& operator = (const PoseMViewPoint &d)
	{
	    yaw = d.yaw;

		for (int i = 0; i < 2*N_POINTS_3; i++)
            points[i] = d.points[i];

		return *this;
	}

	PoseMViewPoint operator + (const PoseMViewPoint &p) const
	{
	    PoseMViewPoint r;

	    r.yaw = yaw + p.yaw;

	    for (int i = 0; i < 2*N_POINTS_3; i++)
            r.points[i] = points[i] + p.points[i];

		return r;
	}

	PoseMViewPoint operator - (const PoseMViewPoint &p) const
	{
	    PoseMViewPoint r;

	    r.yaw = yaw - p.yaw;

	    for (int i = 0; i < 2*N_POINTS_3; i++)
            r.points[i] = points[i] - p.points[i];

		return r;
	}

	PoseMViewPoint operator * (const PoseMViewPoint &p) const
	{
	    PoseMViewPoint r;

	    r.yaw = yaw*p.yaw;

	    for (int i = 0; i < 2*N_POINTS_3; i++)
            r.points[i] = points[i]*p.points[i];

		return r;
	}

	PoseMViewPoint operator / (float p) const
	{
		PoseMViewPoint r;

		r.yaw = yaw/p;

		for (int i = 0; i < 2*N_POINTS_3; i++)
            r.points[i] = points[i]/p;

		return r;
	}

	void setzero()
	{
	    yaw = 0.0f;

	    for (int i = 0; i < 2*N_POINTS_3; i++)
            points[i] = 0.0f;
	}

	static int numvars()
	{
	    return 2*N_POINTS_3 + 1;
	}

	void Pose2Mat(cv::Mat &mat)
    {
        mat = cv::Mat::zeros(1, 2*N_POINTS_3 + 1, CV_64F);

        mat.at<double>(0, 0) = yaw;

        for (int i = 0; i < 2*N_POINTS_3; i++)
            mat.at<double>(0, i + 1) = points[i];
    }

    void Mat2Pose(cv::Mat &mat)
    {
        yaw = mat.at<double>(0, 0);

        for (int i = 0; i < 2*N_POINTS_3; i++)
            points[i] = mat.at<double>(0, i + 1);
    }

    void Pose2Shape(AAM_Shape &shape)
    {
        shape.resize(N_POINTS_3);

        for (int i = 0; i < N_POINTS_3; i++)
        {
            shape[i].x = points[2*i];
            shape[i].y = points[2*i + 1];
        }
    }

    void Shape2Pose(AAM_Shape &shape)
    {
        for (int i = 0; i < N_POINTS_3; i++)
        {
            points[2*i] = shape[i].x;
            points[2*i + 1] = shape[i].y;
        }
    }

	PoseMViewPoint composeWithTarget(const PoseMViewPoint &p) const;
	PoseMViewPoint deltaInTarget(const PoseMViewPoint &p) const;

	static int is_equal(const void* _r1, const void* _r2, void*);
	PoseMViewPoint disturb();

	static PoseMViewPoint calcPose(float* asmshape)
	{
	    int pinds[N_POINTS_3] = {24, 18, 29, 32, 57, 56};

	    PoseMViewPoint pe;

	    pe.yaw = 0.0f;

	    for (int i = 0; i < N_POINTS_3; i++)
	    {
	        pe.points[2*i] = asmshape[2*pinds[i]];
	        pe.points[2*i + 1] = asmshape[2*pinds[i] + 1];
	    }

        return pe;
	}

    double calcCLoss(PoseMViewPoint pe, float* weights, int priority = -1)
    {
        PoseMViewPoint dy = deltaInTarget(pe);
        dy = dy*dy;

        cv::Mat matdy;
        dy.Pose2Mat(matdy);

        int nmult = matdy.cols - 1;

        double loss = 0;

        if (priority == 0)
        {
            loss = loss + matdy.at<double>(0, 0)*weights[0];

            //loss = loss + matdy.at<double>(0, 0)*weights[0]*nmult;

            //for (int i = 1; i < matdy.cols; i++)
                //loss = loss + matdy.at<double>(0, i)*weights[i];
        }
        else
        {
            for (int i = 1; i < matdy.cols; i++)
                loss = loss + matdy.at<double>(0, i)*weights[i];

            //for (int i = 0; i < matdy.cols; i++)
                //loss = loss + matdy.at<double>(0, i)*weights[i];
        }

        return loss;
    }

    double calcMLoss(PoseMViewPoint pe, float* weights, int priority = -1)
    {
        PoseMViewPoint dy = *this - pe;
        dy = dy*dy;

        cv::Mat matdy;
        dy.Pose2Mat(matdy);

        int nmult = matdy.cols - 1;

        double loss = 0;

        if (priority == 0)
        {
            loss = loss + matdy.at<double>(0, 0)*weights[0];

            //loss = loss + matdy.at<double>(0, 0)*weights[0]*nmult;

            //for (int i = 1; i < matdy.cols; i++)
                //loss = loss + matdy.at<double>(0, i)*weights[i];
        }
        else
        {
            for (int i = 1; i < matdy.cols; i++)
                loss = loss + matdy.at<double>(0, i)*weights[i];

            //for (int i = 0; i < matdy.cols; i++)
                //loss = loss + matdy.at<double>(0, i)*weights[i];
        }

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

                pnt.pointidx0 = rand()%N_POINTS_3;
                pnt.pointidx1 = rand()%N_POINTS_3;

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
            cpos.extra0 = rand()%N_POINTS_3;
            cpos.extra1 = rand()%N_POINTS_3;

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
        of << (double)yaw << endl;

        for (int i = 0; i < 2*N_POINTS_3; i++)
            of << (double)points[i] << endl;
    }

    void Read(ifstream &ifs)
    {
        ifs >> yaw;

        for (int i = 0; i < 2*N_POINTS_3; i++)
            ifs >> points[i];
    }
};

void drawOnImage(IplImage* image, PoseMViewPoint &pose, cv::Scalar color = cv::Scalar(255,255,255), int radius = 3, int thickness = 2);

#endif
