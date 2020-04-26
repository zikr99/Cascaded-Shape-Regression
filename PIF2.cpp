#include "PIF2.h"

#define db at<double>
#define SQR(x) (x)*(x)

void CalcSimT(cv::Mat &src, cv::Mat &dst, double &a, double &b, double &tx, double &ty)
{
	assert((src.type() == CV_64F) && (dst.type() == CV_64F) &&
		(src.rows == dst.rows) && (src.cols == dst.cols) && (src.cols == 1));

	int i, n = src.rows/2;
	cv::Mat H(4, 4, CV_64F, cv::Scalar(0));
	cv::Mat g(4, 1, CV_64F, cv::Scalar(0));
	cv::Mat p(4, 1, CV_64F);

	cv::MatIterator_<double> ptr1x = src.begin<double>();
	cv::MatIterator_<double> ptr1y = src.begin<double>() + n;
	cv::MatIterator_<double> ptr2x = dst.begin<double>();
	cv::MatIterator_<double> ptr2y = dst.begin<double>() + n;

	for (i = 0; i < n; i++, ++ptr1x, ++ptr1y, ++ptr2x, ++ptr2y)
	{
		H.db(0, 0) += SQR(*ptr1x) + SQR(*ptr1y);
		H.db(0, 2) += *ptr1x;
		H.db(0, 3) += *ptr1y;

		g.db(0, 0) += (*ptr1x)*(*ptr2x) + (*ptr1y)*(*ptr2y);
		g.db(1, 0) += (*ptr1x)*(*ptr2y) - (*ptr1y)*(*ptr2x);
		g.db(2, 0) += *ptr2x;
		g.db(3, 0) += *ptr2y;
	}

	H.db(1, 1) = H.db(0, 0);
	H.db(1, 2) = H.db(2, 1) = -1.0*(H.db(3, 0) = H.db(0, 3));
	H.db(1, 3) = H.db(3, 1) = H.db(2, 0) = H.db(0, 2);
	H.db(2, 2) = H.db(3, 3) = n;

	cv::solve(H, g, p, CV_CHOLESKY);

	a = p.db(0, 0);
	b = p.db(1, 0);
	tx = p.db(2, 0);
	ty = p.db(3, 0);

	return;
}

void invSimT(double a1, double b1, double tx1, double ty1, double& a2, double& b2, double& tx2, double& ty2)
{
	cv::Mat M = (cv::Mat_<double>(2, 2) << a1, -b1, b1, a1);
	cv::Mat N = M.inv(CV_SVD);

	a2 = N.db(0, 0);
	b2 = N.db(1, 0);
	tx2 = -1.0*(N.db(0, 0)*tx1 + N.db(0, 1)*ty1);
	ty2 = -1.0*(N.db(1, 0)*tx1 + N.db(1, 1)*ty1);

	return;
}

void SimT(cv::Mat &s, double a, double b, double tx, double ty)
{
	assert((s.type() == CV_64F) && (s.cols == 1));

	int i, n = s.rows/2;
	double x, y;

	cv::MatIterator_<double> xp = s.begin<double>(), yp = s.begin<double>() + n;

	for (i = 0; i < n; i++, ++xp, ++yp)
	{
		x = *xp;
		y = *yp;
		*xp = a*x - b*y + tx;
		*yp = b*x + a*y + ty;
	}

	return;
}

cv::Mat getRefPoints(PosePoint* pose)
{
    cv::Mat mrp = (cv::Mat_<double>(6, 1) <<
        pose->points[2*2], pose->points[2*3], pose->points[2*16],
        pose->points[2*2 + 1], pose->points[2*3 + 1], pose->points[2*16 + 1]);

    return mrp;
}

void CalcSimT(PosePoint* pose, double &a, double &b, double &tx, double &ty)
{
    cv::Mat ref = (cv::Mat_<double>(6, 1) <<
        363.943f, 428.908f, 393.798f,
        200.701f, 197.339f, 380.443f);

	cv::Mat pmat = getRefPoints(pose);

	CalcSimT(pmat, ref, a, b, tx, ty);
}

void Pose2MatX(PosePoint* pose, cv::Mat &mat)
{
	mat = cv::Mat::zeros(2*N_POINTS, 1, CV_64F);

	for (int i = 0; i < N_POINTS; i++)
	{
		mat.db(i, 0) = pose->points[2*i];
		mat.db(N_POINTS + i, 0) = pose->points[2*i + 1];
	}
}

void MatX2Pose(cv::Mat &mat, PosePoint* pose)
{
	for (int i = 0; i < N_POINTS; i++)
	{
        pose->points[2*i] = mat.db(i, 0);
        pose->points[2*i + 1] = mat.db(N_POINTS + i, 0);
	}
}

void SimT(PosePoint* s, double a, double b, double tx, double ty)
{
	cv::Mat smat;
	Pose2MatX(s, smat);

	SimT(smat, a, b, tx, ty);

	MatX2Pose(smat, s);
}

PosePoint PosePoint::composeWithTarget(const PosePoint &p) const
{
	double a, b, tx, ty;
	double ai, bi, txi, tyi;

	PosePoint alPose1 = *this;
    CalcSimT(&alPose1, a, b, tx, ty);
	SimT(&alPose1, a, b, tx, ty);

	PosePoint r = alPose1 + p;

	invSimT(a, b, tx, ty, ai, bi, txi, tyi);
	SimT(&r, ai, bi, txi, tyi);

    return r;
}

PosePoint PosePoint::deltaInTarget(const PosePoint &p) const
{
    double a, b, tx, ty;

    PosePoint alPose1 = *this;
    CalcSimT(&alPose1, a, b, tx, ty);
    SimT(&alPose1, a, b, tx, ty);

    PosePoint alPose2 = p;
    SimT(&alPose2, a, b, tx, ty);

    PosePoint r = alPose2 - alPose1;

	return r;
}

int PosePoint::is_equal(const void* _r1, const void* _r2, void*)
{
    const PosePoint* r1 = (const PosePoint*)_r1;
    const PosePoint* r2 = (const PosePoint*)_r2;

    double a, b, tx, ty;

    PosePoint ra1 = *r1;
    CalcSimT(&ra1, a, b, tx, ty);
    SimT(&ra1, a, b, tx, ty);

    PosePoint ra2 = *r2;
    SimT(&ra2, a, b, tx, ty);

    float distance = 20.0f;
    bool iseq = true;

    for (int i = 0; i < N_POINTS; i++)
    {
        float dx = ra1.points[2*i] - ra2.points[2*i];
        float dy = ra1.points[2*i + 1] - ra2.points[2*i + 1];

        if (sqrt(dx*dx + dy*dy) > distance) iseq = false;
    }

    return iseq;
}

PosePoint PosePoint::disturb()
{
    float dx = points[2*7] - points[2*10];
    float dy = points[2*7 + 1] - points[2*10 + 1];
    float dp = sqrt(dx*dx + dy*dy);

    double rot = (2*DIST_POINTS_ROT)*(double)rand()/RAND_MAX - DIST_POINTS_ROT;
    double scale = 1.0f + (2*DIST_POINTS_SCALE)*(double)rand()/RAND_MAX - DIST_POINTS_SCALE;

    double tx = (2*DIST_POINTS_TRANS)*(double)rand()/RAND_MAX - DIST_POINTS_TRANS;
    double ty = (2*DIST_POINTS_TRANS)*(double)rand()/RAND_MAX - DIST_POINTS_TRANS;

    tx = tx*dp;
    ty = ty*dp;

    double sx = points[2*13];
    double sy = points[2*13 + 1];

	double a = scale*cos(rot);
	double b = scale*sin(rot);

	PosePoint pe = *this;

	SimT(&pe, 1.0f, 0.0f, -sx, -sy);
	SimT(&pe, a, b, 0.0f, 0.0f);
    SimT(&pe, 1.0f, 0.0f, sx + tx, sy + ty);

    return pe;
}

void PosePoint::to_img_positions(const vector<DoublePoint> positions, int nFeatures, cv::Point2f* impositions)
{
	double a, b, tx, ty;
	double ai, bi, txi, tyi;
	double x, y;

	PosePoint alPose;

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

void PosePoint::calcFeature(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr)
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

void PosePoint::calcOneFeature(const ImageSample* img, const DoublePoint &position, double &ftr)
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

void PosePoint::calcFeatureComp(const ImageSample* img, const vector<DoublePoint> &positions, int nFeatures, double* ftr)
{
    cv::Point2f *impositions = new cv::Point2f [2*nFeatures];

    to_img_positions(positions, nFeatures, impositions);

    for (int i = 0; i < nFeatures; i++)
        ftr[i] = bilinear_interplotation(&(IplImage)(img->featureChannels[positions[i].channel0]),
            impositions[2*i].x, impositions[2*i].y);

    delete [] impositions;
}

void drawOnImage(IplImage* image, PosePoint &pose, cv::Scalar color, int radius, int thickness)
{
    for (int i = 0; i < N_POINTS; i++)
    {
		cv::Mat matimg(image);
        cv::circle(matimg, cv::Point(pose.points[2*i], pose.points[2*i + 1]), radius, color, thickness);
    }
}

void drawOnImage(IplImage* image, cv::Point2f* points, int npoints, cv::Scalar color, int radius, int thickness)
{
    for (int i = 0; i < npoints; i++)
    {
		cv::Mat matimg(image);
        cv::rectangle(matimg, cv::Rect(points[i].x - radius, points[i].y - radius, 2*radius, 2*radius), color, thickness);
    }
}

