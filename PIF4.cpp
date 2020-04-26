#include "PIF4.h"

#define db at<double>
#define SQR(x) (x)*(x)

cv::Mat getRefPoints(PoseFMViewPoint* pose)
{
    cv::Mat mrp = (cv::Mat_<double>(6, 1) <<
        pose->points[2*2], pose->points[2*3], pose->points[2*18],
        pose->points[2*2 + 1], pose->points[2*3 + 1], pose->points[2*18 + 1]);

    return mrp;
}

void CalcSimT(PoseFMViewPoint* pose, double &a, double &b, double &tx, double &ty)
{
    cv::Mat ref = (cv::Mat_<double>(6, 1) <<
        363.943f, 428.908f, 393.798f,
        200.701f, 197.339f, 380.443f);

	cv::Mat pmat = getRefPoints(pose);

	CalcSimT(pmat, ref, a, b, tx, ty);
}

void Pose2MatX(PoseFMViewPoint* pose, cv::Mat &mat)
{
	mat = cv::Mat::zeros(2*N_POINTS_4, 1, CV_64F);

	for (int i = 0; i < N_POINTS_4; i++)
	{
		mat.db(i, 0) = pose->points[2*i];
		mat.db(N_POINTS_4 + i, 0) = pose->points[2*i + 1];
	}
}

void MatX2Pose(cv::Mat &mat, PoseFMViewPoint* pose)
{
	for (int i = 0; i < N_POINTS_4; i++)
	{
        pose->points[2*i] = mat.db(i, 0);
        pose->points[2*i + 1] = mat.db(N_POINTS_4 + i, 0);
	}
}

void SimT(PoseFMViewPoint* s, double a, double b, double tx, double ty)
{
	cv::Mat smat;
	Pose2MatX(s, smat);

	SimT(smat, a, b, tx, ty);

	MatX2Pose(smat, s);
}

PoseFMViewPoint PoseFMViewPoint::composeWithTarget(const PoseFMViewPoint &p) const
{
	double a, b, tx, ty;
	double ai, bi, txi, tyi;

	PoseFMViewPoint alPose1 = *this;
    CalcSimT(&alPose1, a, b, tx, ty);
	SimT(&alPose1, a, b, tx, ty);

	PoseFMViewPoint r = alPose1 + p;

	invSimT(a, b, tx, ty, ai, bi, txi, tyi);
	SimT(&r, ai, bi, txi, tyi);

    return r;
}

PoseFMViewPoint PoseFMViewPoint::deltaInTarget(const PoseFMViewPoint &p) const
{
    double a, b, tx, ty;

    PoseFMViewPoint alPose1 = *this;
    CalcSimT(&alPose1, a, b, tx, ty);
    SimT(&alPose1, a, b, tx, ty);

    PoseFMViewPoint alPose2 = p;
    SimT(&alPose2, a, b, tx, ty);

    PoseFMViewPoint r = alPose2 - alPose1;

	return r;
}

int PoseFMViewPoint::is_equal(const void* _r1, const void* _r2, void*)
{
    const PoseFMViewPoint* r1 = (const PoseFMViewPoint*)_r1;
    const PoseFMViewPoint* r2 = (const PoseFMViewPoint*)_r2;

    double a, b, tx, ty;

    PoseFMViewPoint ra1 = *r1;
    CalcSimT(&ra1, a, b, tx, ty);
    SimT(&ra1, a, b, tx, ty);

    PoseFMViewPoint ra2 = *r2;
    SimT(&ra2, a, b, tx, ty);

    float distance = 20.0f;
    bool iseq = true;

    for (int i = 0; i < N_POINTS_4; i++)
    {
        float dx = ra1.points[2*i] - ra2.points[2*i];
        float dy = ra1.points[2*i + 1] - ra2.points[2*i + 1];

        if (sqrt(dx*dx + dy*dy) > distance) iseq = false;
    }

    return iseq;
}

PoseFMViewPoint PoseFMViewPoint::disturb()
{
    float dx = points[2*2] - points[2*18];
    float dy = points[2*2 + 1] - points[2*18 + 1];
    float dp1 = sqrt(dx*dx + dy*dy);

    dx = points[2*3] - points[2*18];
    dy = points[2*3 + 1] - points[2*18 + 1];
    float dp2 = sqrt(dx*dx + dy*dy);

    float dp = (dp1 + dp2)/4.0f;

    double rot = (2*DIST_POINTS_ROT_4)*(double)rand()/RAND_MAX - DIST_POINTS_ROT_4;
    double scale = 1.0f + (2*DIST_POINTS_SCALE_4)*(double)rand()/RAND_MAX - DIST_POINTS_SCALE_4;

    double tx = (2*DIST_POINTS_TRANS_4)*(double)rand()/RAND_MAX - DIST_POINTS_TRANS_4;
    double ty = (2*DIST_POINTS_TRANS_4)*(double)rand()/RAND_MAX - DIST_POINTS_TRANS_4;

    tx = tx*dp;
    ty = ty*dp;

    double sx = points[2*14];
    double sy = points[2*14 + 1];

	double a = scale*cos(rot);
	double b = scale*sin(rot);

	PoseFMViewPoint pe = *this;

	SimT(&pe, 1.0f, 0.0f, -sx, -sy);
	SimT(&pe, a, b, 0.0f, 0.0f);
    SimT(&pe, 1.0f, 0.0f, sx + tx, sy + ty);

    return pe;
}

void PoseFMViewPoint::to_img_positions(const vector<DoublePoint> positions, int nFeatures, cv::Point2f* impositions)
{
	double a, b, tx, ty;
	double ai, bi, txi, tyi;
	double x, y;

	PoseFMViewPoint alPose;

	alPose = *this;
	CalcSimT(&alPose, a, b, tx, ty);
	SimT(&alPose, a, b, tx, ty);

	invSimT(a, b, tx, ty, ai, bi, txi, tyi);

	for (int i = 0; i < nFeatures; i++)
	{
		impositions[2*i].x = alPose.points[2*positions[i].pointidx0] + positions[i].x0;
		impositions[2*i].y = alPose.points[2*positions[i].pointidx0 + 1] + positions[i].y0;

		impositions[2*i + 1].x = alPose.points[2*positions[i].pointidx1] + positions[i].x1;
		impositions[2*i + 1].y = alPose.points[2*positions[i].pointidx1 + 1] + positions[i].y1;
	}

	for (int i = 0; i < 2*nFeatures; i++)
	{
		x = impositions[i].x;
		y = impositions[i].y;

		impositions[i].x = ai*x - bi*y + txi;
		impositions[i].y = bi*x + ai*y + tyi;
	}
}

void PoseFMViewPoint::calcFeature(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr)
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

void PoseFMViewPoint::calcOneFeature(const ImageSample* img, const DoublePoint &position, double &ftr)
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

void PoseFMViewPoint::calcFeatureComp(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr)
{
    cv::Point2f *impositions = new cv::Point2f [2*nFeatures];

    to_img_positions(positions, nFeatures, impositions);

    for (int i = 0; i < nFeatures; i++)
        ftr[i] = bilinear_interplotation(&(IplImage)(img->featureChannels[positions[i].channel0]),
            impositions[2*i].x, impositions[2*i].y);

    delete [] impositions;
}

void drawOnImage(IplImage* image, PoseFMViewPoint &pose, cv::Scalar color, int radius, int thickness)
{
    for (int i = 0; i < N_POINTS_4; i++)
    {
		cv::Mat matimg(image);
        cv::circle(matimg, cv::Point(pose.points[2*i], pose.points[2*i + 1]), radius, color, thickness);
    }
}

