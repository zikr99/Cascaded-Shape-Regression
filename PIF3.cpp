#include "PIF3.h"

#define db at<double>
#define SQR(x) (x)*(x)

cv::Mat getRefPoints(PoseMViewPoint* pose)
{
    cv::Mat mrp = (cv::Mat_<double>(6, 1) <<
        pose->points[2*0], pose->points[2*1], pose->points[2*5],
        pose->points[2*0 + 1], pose->points[2*1 + 1], pose->points[2*5 + 1]);

    return mrp;
}

void CalcSimT(PoseMViewPoint* pose, double &a, double &b, double &tx, double &ty)
{
    cv::Mat ref = (cv::Mat_<double>(6, 1) <<
        363.943f, 428.908f, 393.798f,
        200.701f, 197.339f, 380.443f);

	cv::Mat pmat = getRefPoints(pose);

	CalcSimT(pmat, ref, a, b, tx, ty);
}

void Pose2MatX(PoseMViewPoint* pose, cv::Mat &mat)
{
	mat = cv::Mat::zeros(2*N_POINTS_3, 1, CV_64F);

	for (int i = 0; i < N_POINTS_3; i++)
	{
		mat.db(i, 0) = pose->points[2*i];
		mat.db(N_POINTS_3 + i, 0) = pose->points[2*i + 1];
	}
}

void MatX2Pose(cv::Mat &mat, PoseMViewPoint* pose)
{
	for (int i = 0; i < N_POINTS_3; i++)
	{
        pose->points[2*i] = mat.db(i, 0);
        pose->points[2*i + 1] = mat.db(N_POINTS_3 + i, 0);
	}
}

void SimT(PoseMViewPoint* s, double a, double b, double tx, double ty)
{
	cv::Mat smat;
	Pose2MatX(s, smat);

	SimT(smat, a, b, tx, ty);

	MatX2Pose(smat, s);
}

PoseMViewPoint PoseMViewPoint::composeWithTarget(const PoseMViewPoint &p) const
{
	double a, b, tx, ty;
	double ai, bi, txi, tyi;

	PoseMViewPoint alPose1 = *this;
    CalcSimT(&alPose1, a, b, tx, ty);
	SimT(&alPose1, a, b, tx, ty);

	PoseMViewPoint r = alPose1 + p;

	invSimT(a, b, tx, ty, ai, bi, txi, tyi);
	SimT(&r, ai, bi, txi, tyi);

    return r;
}

PoseMViewPoint PoseMViewPoint::deltaInTarget(const PoseMViewPoint &p) const
{
    double a, b, tx, ty;

    PoseMViewPoint alPose1 = *this;
    CalcSimT(&alPose1, a, b, tx, ty);
    SimT(&alPose1, a, b, tx, ty);

    PoseMViewPoint alPose2 = p;
    SimT(&alPose2, a, b, tx, ty);

    PoseMViewPoint r = alPose2 - alPose1;

	return r;
}

int PoseMViewPoint::is_equal(const void* _r1, const void* _r2, void*)
{
    const PoseMViewPoint* r1 = (const PoseMViewPoint*)_r1;
    const PoseMViewPoint* r2 = (const PoseMViewPoint*)_r2;

    double a, b, tx, ty;

    PoseMViewPoint ra1 = *r1;
    CalcSimT(&ra1, a, b, tx, ty);
    SimT(&ra1, a, b, tx, ty);

    PoseMViewPoint ra2 = *r2;
    SimT(&ra2, a, b, tx, ty);

    float distance = 20.0f;
    bool iseq = true;

    for (int i = 0; i < N_POINTS_3; i++)
    {
        float dx = ra1.points[2*i] - ra2.points[2*i];
        float dy = ra1.points[2*i + 1] - ra2.points[2*i + 1];

        if (sqrt(dx*dx + dy*dy) > distance) iseq = false;
    }

    return iseq;
}

PoseMViewPoint PoseMViewPoint::disturb()
{
    float dx = points[2*0] - points[2*5];
    float dy = points[2*0 + 1] - points[2*5 + 1];
    float dp1 = sqrt(dx*dx + dy*dy);

    dx = points[2*1] - points[2*5];
    dy = points[2*1 + 1] - points[2*5 + 1];
    float dp2 = sqrt(dx*dx + dy*dy);

    float dp = (dp1 + dp2)/2.0f;

    double rot = (2*DIST_POINTS_ROT)*(double)rand()/RAND_MAX - DIST_POINTS_ROT;
    double scale = 1.0f + (2*DIST_POINTS_SCALE)*(double)rand()/RAND_MAX - DIST_POINTS_SCALE;

    double tx = (2*DIST_POINTS_TRANS)*(double)rand()/RAND_MAX - DIST_POINTS_TRANS;
    double ty = (2*DIST_POINTS_TRANS)*(double)rand()/RAND_MAX - DIST_POINTS_TRANS;

    tx = tx*dp;
    ty = ty*dp;

    double sx = points[2*4];
    double sy = points[2*4 + 1];

	double a = scale*cos(rot);
	double b = scale*sin(rot);

	PoseMViewPoint pe = *this;

	SimT(&pe, 1.0f, 0.0f, -sx, -sy);
	SimT(&pe, a, b, 0.0f, 0.0f);
    SimT(&pe, 1.0f, 0.0f, sx + tx, sy + ty);

    return pe;
}

void PoseMViewPoint::to_img_positions(const vector<DoublePoint> positions, int nFeatures, cv::Point2f* impositions)
{
	double a, b, tx, ty;
	double ai, bi, txi, tyi;
	double x, y;

	PoseMViewPoint alPose;

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

void PoseMViewPoint::calcFeature(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr)
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

void PoseMViewPoint::calcOneFeature(const ImageSample* img, const DoublePoint &position, double &ftr)
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

void PoseMViewPoint::calcFeatureComp(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr)
{
    cv::Point2f *impositions = new cv::Point2f [2*nFeatures];

    to_img_positions(positions, nFeatures, impositions);

    for (int i = 0; i < nFeatures; i++)
        ftr[i] = bilinear_interplotation(&(IplImage)(img->featureChannels[positions[i].channel0]),
            impositions[2*i].x, impositions[2*i].y);

    delete [] impositions;
}

void drawOnImage(IplImage* image, PoseMViewPoint &pose, cv::Scalar color, int radius, int thickness)
{
    for (int i = 0; i < N_POINTS_3; i++)
    {
		cv::Mat matimg(image);
        cv::circle(matimg, cv::Point(pose.points[2*i], pose.points[2*i + 1]), radius, color, thickness);
    }
}

