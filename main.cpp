#include <Windows.h>
#include <direct.h>
#include <cv.h>

#include "aflw/dbconn/SQLiteDBConnection.h"
#include "aflw/facedata/FaceData.h"
#include "aflw/querys/FaceDataByIDsQuery.h"
#include "aflw/util/MeanFace3DModel.h"
#include "aflw/util/ModernPosit.h"

#include "ASMDLL/ASMDLL.h"
#include "image_sample.hpp"
#include "debug.h"
#include "PIF4.h"
#include "RandomFern.h"
#include "tree/tree.hpp"
#include "CPR2.h"

#define DB_DIR "C:\\AFLW\\data"
#define DB_FILE "C:\\AFLW\\data\\aflw.sqlite"

using namespace cv;

const std::string G_FeatureCodes[21] = {"LBLC", "LBC", "LBRC", "RBLC", "RBC", "RBRC", "LELC", "LEC", "LERC", "RELC",
    "REC", "RERC", "LE", "LN", "NC", "RN", "RE", "MLC", "MC", "MRC", "CC"};

FeatureCoordTypes G_fcTypes;
MeanFace3DModel G_meanFace3DModel;

void InitPOSIT()
{
    SQLiteDBConnection conn;
	conn.open(DB_FILE);

    G_fcTypes.load(&conn);
    G_meanFace3DModel.load(&conn);

    conn.close();
}

void clearAnnotations(vector<FaceData*> &annotations)
{
    for (int i = 0; i < annotations.size(); i++)
		delete annotations.at(i);

    annotations.clear();
}

FaceData* retrieveFaceData(int queryId, SQLiteDBConnection* conn)
{
    vector<int> queryIds;
    queryIds.push_back(queryId);

	FaceDataByIDsQuery faceDataSqlQuery;
	faceDataSqlQuery.queryIds = queryIds;

	FaceData* ret_annotations = NULL;

	bool allOk = true;
	allOk = allOk && faceDataSqlQuery.exec(conn);

	if (!allOk)
	{
		cout << "There's an error !" << endl;
		return ret_annotations;
	}

	std::map<int, FaceData*> annotations = faceDataSqlQuery.data;
    FaceData *currFaceAnnotation = annotations[queryIds.at(0)];

    //FaceData has to be copied as it is destroyed on destruction of the query
    FaceData *cpyCurrFaceAnno = new FaceData(*currFaceAnnotation);
    ret_annotations = cpyCurrFaceAnno;

	return ret_annotations;
}

void splitRotationMatrixToRollPitchYaw(cv::Mat &rot_matrix, double& roll, double& pitch, double& yaw)
{
    // do we have to transpose here?
    const double a11 = rot_matrix.at<float>(0,0), a12 = rot_matrix.at<float>(0,1), a13 = rot_matrix.at<float>(0,2);
    const double a21 = rot_matrix.at<float>(1,0), a22 = rot_matrix.at<float>(1,1), a23 = rot_matrix.at<float>(1,2);
    const double a31 = rot_matrix.at<float>(2,0), a32 = rot_matrix.at<float>(2,1), a33 = rot_matrix.at<float>(2,2);

    //replaces vnl_math header
    const double epsilon = 2.2204460492503131e-16;
    double pi = 3.14159265358979323846;
    double pi_over_2 = 1.57079632679489661923;
    double pi_over_4 = 0.78539816339744830962;

    if (abs(1.0 - a31) <= epsilon) // special case a31 == +1
    {
        //qDebug() << "gimbal lock case a31 == " << a31;
        pitch = -pi_over_2;
        yaw   = pi_over_4; // arbitrary value
        roll  = atan2(a12,a13) - yaw;
    }
    else if (abs(-1.0 - a31) <= epsilon) // special case a31 == -1
    {
        //qDebug() << "gimbal lock case a31 == ";
        pitch = pi_over_2;
        yaw   = pi_over_4; // arbitrary value
        roll  = atan2(a12,a13) + yaw;
    }
    else // standard case a31 != +/-1
    {
        pitch = asin(-a31);

        //two cases depending on where pitch angle lies
        if ((pitch < pi_over_2) && (pitch > -pi_over_2))
        {
            roll = atan2(a32,a33);
            yaw  = atan2(a21,a11);
        }
        else if ((pitch < 3.0 * pi_over_2) && (pitch > pi_over_2))
        {
            roll = atan2(-a32,-a33);
            yaw  = atan2(-a21,-a11);
        }
        else
        {
            std::cerr << "this should never happen in pitch roll yaw computation!" << std::endl;
            roll = 2.0*pi;
            yaw  = 2.0*pi;
        }
    }
}

PoseFMViewPoint getPoseComplete(PoseFMViewPoint &pose, cv::Mat &img, bool *exfts)
{
    int availableFeatureIds[21] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    double focusFactor = 1.50f;

    ModernPosit modernPosit;
	cv::Mat rot;
	cv::Point3f trans;

	double focalLength = static_cast<double>(img.cols)*1.5f;
	cv::Point2f imgCenter = cv::Point2f(static_cast<float>(img.cols)/2.0f, static_cast<float>(img.rows)/2.0f);

	std::vector<cv::Point2f> imagePts;
	std::vector<cv::Point3f> worldPts;

	for (int i = 0; i < 21; i++)
        if (exfts[i])
        {
            cv::Point2f imgPt = cv::Point2f(pose.points[2*i], pose.points[2*i + 1]);
            int featureID = availableFeatureIds[i];

            if ((imgPt.x >= 0) && (imgPt.y >= 0))
            {
                imagePts.push_back(imgPt);

                std::string featureName = G_fcTypes.getCode(featureID);
                cv::Point3f worldPt = G_meanFace3DModel.getCoordsByCode(featureName);

                // adjust the world and model points to move them from opengl to pitch yaw roll coordinate system
                cv::Point3f worldPtOgl;
                worldPtOgl.x = worldPt.z;
                worldPtOgl.y = -worldPt.x;
                worldPtOgl.z = -worldPt.y;

                worldPts.push_back(worldPtOgl);
            }
        }

	modernPosit.run(rot, trans, imagePts, worldPts, focalLength, imgCenter);

	cv::Mat m_tfm = (cv::Mat_<float>(4,4) << 0,0,0,trans.x, 0,0,0,trans.y, 0,0,0,trans.z, 0,0,0,1);

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			m_tfm.at<float>(i, j) = rot.at<float>(i, j);

	cv::Mat rotateCoordSystemM = (cv::Mat_<float>(4, 4) << 0,0,-1,0, -1,0,0,0, 0,1,0,0, 0,0,0,1);
	cv::Mat tmp = rotateCoordSystemM*m_tfm;

    double croll, cpitch, cyaw;
	splitRotationMatrixToRollPitchYaw(tmp, croll, cpitch, cyaw);

    const double center_x = 0.5f*img.cols;
    const double center_y = 0.5f*img.rows;
    const double distanceToImagePlane = focusFactor*img.cols;

    //std::cout << "TransformedFeaturePointsOnly" << m_tfm << std::endl;
    //Project the model points with the estimated pose

    PoseFMViewPoint pe;

    for (int counter = 0; counter < 21; counter++)
	{
        cv::Point3f pt3d = G_meanFace3DModel.getCoordsByCode(G_FeatureCodes[counter]);
		cv::Mat mPt3d = (cv::Mat_<float>(4,1) << pt3d.z, -pt3d.x, -pt3d.y, 1); //OGL coords...

		mPt3d = m_tfm*mPt3d; //rotat and translate model in 3D

		if (mPt3d.at<float>(2,0) != 0)
		{
			pe.points[2*counter] = distanceToImagePlane*mPt3d.at<float>(0, 0)/mPt3d.at<float>(2,0); //project to 2D
			pe.points[2*counter + 1] = distanceToImagePlane*mPt3d.at<float>(1, 0)/mPt3d.at<float>(2,0);

			pe.points[2*counter] = pe.points[2*counter] + center_x;
			pe.points[2*counter + 1] = pe.points[2*counter + 1] + center_y;
		}
    }

    for (int counter = 0; counter < 21; counter++)
        if (exfts[counter])
        {
            pe.points[2*counter] = pose.points[2*counter];
			pe.points[2*counter + 1] = pose.points[2*counter + 1];
        }

    pe.yaw = pose.yaw;

    return pe;
}

bool getPoseFromCoords(FeaturesCoords* fccoords, double yaw, cv::Mat &img, PoseFMViewPoint &cppose)
{
    bool exfts[22];
    for (int i = 0; i < 22; i++) exfts[i] = false;

    vector<int> fids = fccoords->getFeatureIds();
    for (int i = 0; i < fids.size(); i++) exfts[fids[i]] = true;

    bool chk;

    if ((yaw >= 2*CV_PI/6.0f) && (yaw <= 3*CV_PI/6.0f))
    {
        chk = true;

        if (!exfts[4] && !exfts[5] && !exfts[6] && !exfts[10] && !exfts[11] && !exfts[12]) chk = false;
        if (!exfts[15]) chk = false;
        if (!exfts[17]) chk = false;
        if (!exfts[21]) chk = false;
    }
    else if ((yaw >= 1*CV_PI/6.0f) && (yaw <= 2*CV_PI/6.0f))
    {
        chk = true;

        if (!exfts[4] && !exfts[5] && !exfts[6] && !exfts[10] && !exfts[11] && !exfts[12]) chk = false;
        if (!exfts[15]) chk = false;
        if (!exfts[21]) chk = false;
        if (!exfts[1] && !exfts[2] && !exfts[3] && !exfts[7] && !exfts[8] && !exfts[9]) chk = false;
    }
    else if ((yaw >= -1*CV_PI/6.0f) && (yaw <= 1*CV_PI/6.0f))
    {
        chk = true;

        if (!exfts[1] && !exfts[2] && !exfts[3] && !exfts[7] && !exfts[8] && !exfts[9]) chk = false;
        if (!exfts[4] && !exfts[5] && !exfts[6] && !exfts[10] && !exfts[11] && !exfts[12]) chk = false;
        if (!exfts[14] && !exfts[15] && !exfts[16] && !exfts[18] && !exfts[19] && !exfts[20] && !exfts[21]) chk = false;
    }
    else if ((yaw >= -2*CV_PI/6.0f) && (yaw <= -1*CV_PI/6.0f))
    {
        chk = true;

        if (!exfts[1] && !exfts[2] && !exfts[3] && !exfts[7] && !exfts[8] && !exfts[9]) chk = false;
        if (!exfts[15]) chk = false;
        if (!exfts[21]) chk = false;
        if (!exfts[4] && !exfts[5] && !exfts[6] && !exfts[10] && !exfts[11] && !exfts[12]) chk = false;
    }
    else if ((yaw >= -3*CV_PI/6.0f) && (yaw <= -2*CV_PI/6.0f))
    {
        chk = true;

        if (!exfts[1] && !exfts[2] && !exfts[3] && !exfts[7] && !exfts[8] && !exfts[9]) chk = false;
        if (!exfts[15]) chk = false;
        if (!exfts[13]) chk = false;
        if (!exfts[21]) chk = false;
    }
    else
    {
        chk = false;
    }

    if (!chk) return false;

    vector<cv::Point2f> points1;

    if ((yaw >= 2*CV_PI/6.0f) && (yaw <= 3*CV_PI/6.0f))
    {
        cppose.yaw = yaw;

        for (int i = 1; i < 22; i++)
        {
            cv::Point2f cpt;

            if ((i == 1) || (i == 2))
                cpt = fccoords->getCoords(exfts[i]?i:(exfts[3]?3:7 - i));
            else if (i == 3)
                cpt = fccoords->getCoords(exfts[3]?3:4);
            else if ((i == 7) || (i == 9))
                cpt = fccoords->getCoords(exfts[i]?i:(exfts[8]?8:19 - i));
            else if (i == 8)
                cpt = fccoords->getCoords(exfts[8]?8:11);
            else if (i == 14)
                cpt = fccoords->getCoords(exfts[14]?14:16);
            else if (i == 18)
                cpt = fccoords->getCoords(exfts[18]?18:20);
            else
                cpt = fccoords->getCoords(i);

            points1.push_back(cpt);
        }
    }
    else if ((yaw >= 1*CV_PI/6.0f) && (yaw <= 2*CV_PI/6.0f))
    {
        cppose.yaw = yaw;

        for (int i = 1; i < 22; i++)
        {
            cv::Point2f cpt;

            if ((i == 1) || (i == 2))
                cpt = fccoords->getCoords(exfts[i]?i:3);
            else if ((i == 7) || (i == 9))
                cpt = fccoords->getCoords(exfts[i]?i:(exfts[8]?8:19 - i));
            else if (i == 8)
                cpt = fccoords->getCoords(exfts[8]?8:11);
            else if (i == 14)
                cpt = fccoords->getCoords(exfts[14]?14:15);
            else
                cpt = fccoords->getCoords(i);

            points1.push_back(cpt);
        }
    }
    else if ((yaw >= -1*CV_PI/6.0f) && (yaw <= 1*CV_PI/6.0f))
    {
        cppose.yaw = yaw;

        for (int i = 1; i < 22; i++)
        {
            cv::Point2f cpt = fccoords->getCoords(i);
            points1.push_back(cpt);
        }
    }
    else if ((yaw >= -2*CV_PI/6.0f) && (yaw <= -1*CV_PI/6.0f))
    {
        cppose.yaw = yaw;

        for (int i = 1; i < 22; i++)
        {
            cv::Point2f cpt;

            if ((i == 5) || (i == 6))
                cpt = fccoords->getCoords(exfts[i]?i:4);
            else if ((i == 10) || (i == 12))
                cpt = fccoords->getCoords(exfts[i]?i:(exfts[11]?11:19 - i));
            else if (i == 11)
                cpt = fccoords->getCoords(exfts[11]?11:8);
            else if (i == 16)
                cpt = fccoords->getCoords(exfts[16]?16:15);
            else
                cpt = fccoords->getCoords(i);

            points1.push_back(cpt);
        }
    }
    else if ((yaw >= -3*CV_PI/6.0f) && (yaw <= -2*CV_PI/6.0f))
    {
        cppose.yaw = yaw;

        for (int i = 1; i < 22; i++)
        {
            cv::Point2f cpt;

            if ((i == 5) || (i == 6))
                cpt = fccoords->getCoords(exfts[i]?i:(exfts[4]?4:7 - i));
            else if (i == 4)
                cpt = fccoords->getCoords(exfts[4]?4:3);
            else if ((i == 10) || (i == 12))
                cpt = fccoords->getCoords(exfts[i]?i:(exfts[11]?11:19 - i));
            else if (i == 11)
                cpt = fccoords->getCoords(exfts[11]?11:8);
            else if (i == 16)
                cpt = fccoords->getCoords(exfts[16]?16:14);
            else if (i == 20)
                cpt = fccoords->getCoords(exfts[20]?20:18);
            else
                cpt = fccoords->getCoords(i);

            points1.push_back(cpt);
        }
    }

    for (int i = 0; i < points1.size(); i++)
    {
        cppose.points[2*i] = points1[i].x;
        cppose.points[2*i + 1] = points1[i].y;
    }

    cppose = getPoseComplete(cppose, img, &exfts[1]);

    return true;
}

void LoadData(char* filename, vector<string> &imgnames, vector<PoseFMViewPoint> &truePose, vector<PoseFMViewPoint> &startPose)
{
	ifstream ifs(filename);

	int sz;
	ifs >> sz;

	vector<int> faceids(sz);

	for (int i = 0; i < sz; i++)
        ifs >> faceids[i];

	ifs.close();

	SQLiteDBConnection conn;
	conn.open(DB_FILE);

    for (int i = 0; i < faceids.size(); i++)
    {
        FaceData* my_annotation = retrieveFaceData(faceids[i], &conn);

        my_annotation->loadFeatureCoords(&conn);
        FeaturesCoords *fccoords = my_annotation->getFeaturesCoords();
        FacePose *fcpose = my_annotation->getPose();

        char imgfpath[400];

        sprintf(imgfpath, "%s/%s%s", DB_DIR, my_annotation->getDbImg()->dbpath.c_str(),
            my_annotation->getDbImg()->filepath.c_str());

        string stringpath = string(imgfpath);
        cv::Mat timg = cv::imread(stringpath, 1);

        PoseFMViewPoint cppose;
        cppose.setzero();

        if (getPoseFromCoords(fccoords, fcpose->yaw, timg, cppose))
        {
            imgnames.push_back(stringpath);
            truePose.push_back(cppose);


            drawOnImage(&(IplImage)timg, cppose, cv::Scalar(0, 0, 255));


            my_annotation->loadRects(&conn);
            vector<FaceRect> rects = my_annotation->getRects();

            int numpoints;
            float *asmshape;
            CvRect rect;

            rect.x = rects[0].x;
            rect.y = rects[0].y;
            rect.width = rects[0].w;
            rect.height = rects[0].h;

            cv::Mat dump = cv::Mat::zeros(100, 100, CV_8UC3);
            ASMDLLFit(0, numpoints, &asmshape, &(IplImage)dump, &rect, true);

            cppose = PoseFMViewPoint::calcPose(asmshape);
            startPose.push_back(cppose);

            FreeFitResult(&asmshape);


            drawOnImage(&(IplImage)timg, cppose, cv::Scalar(0, 255, 0));

            char outfimg[200];
            sprintf(outfimg, "tests\\%d.jpg", i);
            imwrite(outfimg, timg);


            cout << "load-" << imgnames.size() - 1 << endl;
        }

        delete my_annotation;
    }

    conn.close();
}

void LoadDataOldComp(char* filename, vector<string> &images, vector<PoseFMViewPoint> &truePose, vector<PoseFMViewPoint> &startPose)
{
	ifstream ifs(filename);

	int sz;
	ifs >> sz; cout << sz << endl;

    for (int i = 0; i < sz; i++)
    {
        string imstring;
        ifs >> imstring;

        char imgfpath[400];
        sprintf(imgfpath, "%s", imstring.c_str());
        imgfpath[36] = '\0';

        cout << "load-" << i << ": " << imgfpath << endl;
        images.push_back(imgfpath);

        PoseFMViewPoint pose;
        pose.Read(ifs);
        truePose.push_back(pose);

        pose.Read(ifs);
        startPose.push_back(pose);
    }

    ifs.close();
}

void LoadDataTest(char* filename, vector<string> &imgnames, vector<PoseFMViewPoint> &truePose,
    vector<vector<bool> > &ptflags, vector<double> &normds, vector<PoseFMViewPoint> &startPose)
{
    int inds[21] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    bool exfts[22];

	ifstream ifs(filename);

	int sz;
	ifs >> sz;

	vector<int> faceids(sz);

	for (int i = 0; i < sz; i++)
        ifs >> faceids[i];

	ifs.close();

	SQLiteDBConnection conn;
	conn.open(DB_FILE);

    for (int i = 0; i < faceids.size(); i++)
    {
        FaceData* my_annotation = retrieveFaceData(faceids[i], &conn);

        my_annotation->loadFeatureCoords(&conn);
        FeaturesCoords *fccoords = my_annotation->getFeaturesCoords();
        FacePose *fcpose = my_annotation->getPose();

        char imgfpath[400];

        sprintf(imgfpath, "%s/%s%s", DB_DIR, my_annotation->getDbImg()->dbpath.c_str(),
            my_annotation->getDbImg()->filepath.c_str());

        for (int j = 0; j < 22; j++) exfts[j] = false;

        vector<int> fids = fccoords->getFeatureIds();
        for (int j = 0; j < fids.size(); j++) exfts[fids[j]] = true;

        PoseFMViewPoint pose;
        pose.setzero();

        vector<bool> cflags;
        cflags.resize(21);

        for (int j = 0; j < 21; j++)
            if (exfts[inds[j]])
            {
                pose.points[2*j] = fccoords->getCoords(inds[j]).x;
                pose.points[2*j + 1] = fccoords->getCoords(inds[j]).y;

                cflags[j] = true;
            }
            else
            {
                cflags[j] = false;
            }

        double dx, dy;
        double d1, d2, dd;

        if (exfts[inds[2]] && exfts[inds[18]])
        {
            dx = pose.points[2*2] - pose.points[2*18];
            dy = pose.points[2*2 + 1] - pose.points[2*18 + 1];

            d1 = sqrt(dx*dx + dy*dy);
        }

        if (exfts[inds[3]] && exfts[inds[18]])
        {
            dx = pose.points[2*3] - pose.points[2*18];
            dy = pose.points[2*3 + 1] - pose.points[2*18 + 1];

            d2 = sqrt(dx*dx + dy*dy);
        }

        if (exfts[inds[2]] && exfts[inds[3]] && exfts[inds[18]])
            dd = (d1 + d2)/2.0f;
        else if (exfts[inds[2]] && exfts[inds[18]])
            dd = d1;
        else if (exfts[inds[3]] && exfts[inds[18]])
            dd = d2;
        else
            continue;

        string stringpath = string(imgfpath);
        imgnames.push_back(stringpath);

        truePose.push_back(pose);
        ptflags.push_back(cflags);

        normds.push_back(dd);

        my_annotation->loadRects(&conn);
        vector<FaceRect> rects = my_annotation->getRects();

        int numpoints;
        float *asmshape;
        CvRect rect;

        rect.x = rects[0].x;
        rect.y = rects[0].y;
        rect.width = rects[0].w;
        rect.height = rects[0].h;

        cv::Mat dump = cv::Mat::zeros(100, 100, CV_8UC3);
        ASMDLLFit(0, numpoints, &asmshape, &(IplImage)dump, &rect, true);

        PoseFMViewPoint cppose = PoseFMViewPoint::calcPose(asmshape);
        startPose.push_back(cppose);

        FreeFitResult(&asmshape);

        cout << "load-" << imgnames.size() - 1 << endl;

        delete my_annotation;
    }

    conn.close();
}

int main()
{
    srand(time(0));

    InitASMDLL(1);
    SetASMDLL(0, "ASMAAM\\asm2.amf", "ASMAAM\\haarcascade2.xml");

    bool trainmode = true;

    vector<string> images;
	vector<PoseFMViewPoint> truePose0;
	vector<PoseFMViewPoint> startPose0;
	vector<PoseFMViewPoint> truePose;
	vector<PoseFMViewPoint> startPose;
	vector<vector<bool> > ptflags;
	vector<double> normds;

    if (trainmode)
    {
        LoadDataOldComp("Tr-3.txt", images, truePose0, startPose0);
        cout << "Train" << endl;

        truePose.resize(images.size());
        startPose.resize(images.size());

        for (int i = 0; i < images.size(); i++)
        {
            startPose[i] = startPose0[i];
            truePose[i] = truePose0[i];
        }

        int nSample = images.size();
        int nAugment = 11;
        int nStage = 1000;
        int nStagePart = 30;
        int nFeature = 400;

        RandomFernParam param;
        param.nDepth = 8;
        param.beta = 40;
        param.nTry = 400;

        int lbpriorities[1500];
        for (int i = 0; i < 1500; i++) lbpriorities[i] = -1;

		CPR<DoublePoint, RandomFern, PoseFMViewPoint, RandomFernParam> cpr;
        CPR<DoublePoint, RandomFern, PoseFMViewPoint, RandomFernParam>::create_cpr(&cpr, nStage, param);

        cpr.cpr_train(images, truePose, startPose, nSample, nAugment, nStage, nStagePart, nFeature, param, lbpriorities);
        CPR<DoublePoint, RandomFern, PoseFMViewPoint, RandomFernParam>::save_cpr(&cpr, "pose1000.txt");

        CPR<DoublePoint, RandomFern, PoseFMViewPoint, RandomFernParam>::release_cpr(&cpr);
    }
    else
    {
        LoadDataTest("pose_test.txt", images, truePose, ptflags, normds, startPose);

		CPR<DoublePoint, RandomFern, PoseFMViewPoint, RandomFernParam> cpr;
		CPR<DoublePoint, RandomFern, PoseFMViewPoint, RandomFernParam>::load_cpr(&cpr, "pose200-0.txt");
		int nCluster = 16;

        ofstream tsfs("..\\experiments\\point-coup.txt");
        ofstream fgfs("..\\experiments\\point-coup-flags.txt");

		for (int i = 0; i < truePose.size(); i++)
		{
		    cout << i << endl;

			PoseFMViewPoint predict_pose;
			cpr.fit(images[i], startPose[i], predict_pose, nCluster);

			PoseFMViewPoint errpose = truePose[i] - predict_pose;

			for (int j = 0; j < 19; j++)
			{
                double dx = errpose.points[2*j];
                double dy = errpose.points[2*j + 1];

                tsfs << sqrt(dx*dx + dy*dy)/normds[i] << " ";
			}

			tsfs << endl;

			for (int j = 0; j < 19; j++)
                if (ptflags[i][j])
                    fgfs << "1 ";
                else
                    fgfs << "0 ";

			fgfs << endl;

			cv::Mat outimg = imread(images[i], 1);
			drawOnImage(&(IplImage)outimg, truePose[i], cv::Scalar(0, 0, 255));
			drawOnImage(&(IplImage)outimg, predict_pose, cv::Scalar(0, 255, 0));

			char outfname[200];
			sprintf(outfname, "tests\\%d.jpg", i);
            cv::imwrite(outfname, outimg);
		}

		tsfs.close();
		fgfs.close();

		CPR<DoublePoint, RandomFern, PoseFMViewPoint, RandomFernParam>::release_cpr(&cpr);
	}

	FinalASMDLL();
	system("pause");
}
