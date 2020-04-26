#ifndef CASCADE_POSE_REGRESSION
#define CASCADE_POSE_REGRESSION

#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <vector>
#include <stdio.h>

#include "debug.h"
#include "PIF.h"
#include "RandomFern.h"
#include "AAMLibrary/AAM_PDM.h"
#include "AAMLibrary/AAM_Shape.h"

using namespace std;

template<class PoseParameter>
void calcDWeights(PoseParameter* deltaTarget, int nSample, float* weights)
{
    PoseParameter vacc;

	vacc.setzero();

	for (int i = 0; i < nSample; i++)
		vacc = vacc + deltaTarget[i]*deltaTarget[i];

	vacc = vacc/nSample;

    cv::Mat matvacc;
	vacc.Pose2Mat(matvacc);

	for (int i = 0; i < matvacc.cols; i++)
		weights[i] = 1.0f/sqrt(matvacc.at<double>(0, i));
}

template<class PoseParameter>
void calcVWeights(PoseParameter* poses, int nSample, float* weights)
{
    PoseParameter mean;
    PoseParameter vacc;

    mean.setzero();

    for (int i = 0; i < nSample; i++)
		mean = mean + poses[i];

    mean = mean/nSample;

	vacc.setzero();

	for (int i = 0; i < nSample; i++)
		vacc = vacc + (poses[i] - mean)*(poses[i] - mean);

	vacc = vacc/nSample;

    cv::Mat matvacc;
	vacc.Pose2Mat(matvacc);

	for (int i = 0; i < matvacc.cols; i++)
		weights[i] = 1.0f/sqrt(matvacc.at<double>(0, i));
}

template<class FeatureType, template<class PoseParameter> class TreeModel, class PoseParameter, class TreeParam>
struct StageRegression
{
    int r;
    vector<FeatureType> features;
	TreeModel<PoseParameter> tree;

	static void create_stage_regression(StageRegression* sr, TreeParam param)
    {
	    sr->features.clear();
        TreeModel<PoseParameter>::create(&sr->tree, param);
    }

    static void release_stage_regression(StageRegression* sr)
    {
	    sr->features.clear();
	    TreeModel<PoseParameter>::release(&sr->tree);
    }

    void Write(ofstream &of)
    {
        of << (int)r << endl;
        of << (int)features.size() << endl;

        for (int i = 0; i < features.size(); i++)
            features[i].Write(of);

        tree.Write(of);
    }

    void Read(ifstream &ifs)
    {
        ifs >> r;

        int nfet;
        ifs >> nfet;

        features.resize(nfet);

        for (int i = 0; i < nfet; i++)
            features[i].Read(ifs);

        tree.Read(ifs);
    }
};

template<class FeatureType, template<class PoseParameter> class TreeModel, class PoseParameter, class TreeParam>
struct CPR
{
	int nStage;
	StageRegression<FeatureType, TreeModel, PoseParameter, TreeParam>* sr;

	CPR()
	{
	    nStage = 0;
	    sr = 0;
	}

	static void create_cpr(CPR *cpr, int nStage, TreeParam param)
    {
        cpr->nStage = nStage;
        cpr->sr = new StageRegression<FeatureType, TreeModel, PoseParameter, TreeParam> [nStage];

        for (int i = 0; i < nStage; i++)
            StageRegression<FeatureType, TreeModel, PoseParameter, TreeParam>::create_stage_regression(&cpr->sr[i], param);
    }

    static void release_cpr(CPR* cpr)
    {
        for (int i = 0; i < cpr->nStage; i++)
            StageRegression<FeatureType, TreeModel, PoseParameter, TreeParam>::release_stage_regression(&cpr->sr[i]);

        delete [] cpr->sr;
    }

    void cpr_train(vector<ImageSample> &images, vector<PoseParameter> &_truePose, vector<PoseParameter> &_startPose,
        int nSample, int nAugment, int nStage, int nStagePart, int nFeature, TreeParam param)
    {
        PoseParameter *truePose = new PoseParameter [nSample*nAugment];
        PoseParameter *curPose = new PoseParameter [nSample*nAugment];

        int* imIndex = new int [nSample*nAugment];

        for (int i = 0; i < nAugment; i++)
        {
            for (int j = 0; j < nSample; j++)
            {
                imIndex[i*nSample + j] = j;
                truePose[i*nSample + j] = _truePose[j];

                if (i == 0)
                    curPose[i*nSample + j] = _startPose[j];
                else
                    curPose[i*nSample + j] = _startPose[j].disturb();
            }
        }

        //learning loop

        nSample = nAugment*nSample;
        int nVars = PoseParameter::numvars();

        PoseParameter *refcurPose = new PoseParameter [nSample*nAugment];
        PoseParameter *yPose = new PoseParameter [nSample*nAugment];

        for (int i = 0; i < nSample; i++)
            yPose[i] = curPose[i].deltaInTarget(truePose[i]);

        float *weights = new float [nVars];
        calcDWeights(yPose, nSample, weights);

        double curLoss = 0;

        for (int i = 0; i < nSample; i++)
            curLoss = curLoss + curPose[i].calcCLoss(truePose[i], weights);

        curLoss = curLoss/nSample;
        printf("curLoss: %lf\n", curLoss);

        vector<FeatureType> positions;

        double** f = new double* [nSample];
        for (int i = 0; i < nSample; i++) f[i] = new double [nFeature];

        int *lvindices = new int [nSample];

        for (int t = 0; t < nStage; t++)
        {
            cout << "Stage: " << t << endl;

            if (t%nStagePart == 0)
                for (int i = 0; i < nSample; i++) refcurPose[i] = curPose[i];

            PoseParameter::generate_random_positions(positions, nFeature);

            for (int i = 0; i < nSample; i++)
            {
                refcurPose[i].calcFeature(&images[imIndex[i]], positions, nFeature, f[i]);
                yPose[i] = curPose[i].deltaInTarget(truePose[i]);
            }

            TreeModel<PoseParameter> tree;
            TreeModel<PoseParameter>::create(&tree, param);

            tree.train(f, nSample, nFeature, yPose, param, lvindices);

            //save random fern to cpr
            StageRegression<FeatureType, TreeModel, PoseParameter, TreeParam>::create_stage_regression(&sr[t], param);

            sr[t].r = t - t%nStagePart;
            sr[t].features = positions;
            sr[t].tree = tree;

            TreeModel<PoseParameter>::release(&tree);

            for (int i = 0; i < nSample; i++)
            {
                curPose[i] = curPose[i].composeWithTarget(sr[t].tree.regress(lvindices[i]));
            }

            curLoss = 0;

            for (int i = 0; i < nSample; i++)
                curLoss = curLoss + curPose[i].calcCLoss(truePose[i], weights);

            curLoss = curLoss/nSample;
            printf("curLoss: %lf\n", curLoss);
        }

        for (int i = 0; i < nSample; i++)
            delete [] f[i];

        delete [] f;
        delete [] lvindices;

        delete [] refcurPose;
        delete [] yPose;
        delete [] weights;

        delete [] truePose;
        delete [] curPose;
        delete [] imIndex;
    }

    void fit(ImageSample &image, PoseParameter &startPose, PoseParameter &predictPose, int nCluster)
    { static int ididx = 0; ididx++;
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvMemStorage* storage_index = cvCreateMemStorage(0);
        CvSeq *seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(PoseParameter), storage);
        CvSeq *index_seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(int), storage_index);

        PoseParameter* initialPose = new PoseParameter [nCluster];

        for (int i = 0; i < nCluster; i++)
            initialPose[i] = startPose.disturb();

        PoseParameter curPose, refcurPose, predictpose;

        for (int i = 0; i < nCluster; i++)
        {
            curPose = initialPose[i];

            for (int t = 0; t < nStage; t++)
            {
                if ((i == 0) && (t < 300))
                {
                    cv::Mat outimg = image.featureChannels[0].clone();

                    char imgname[200];
                    sprintf(imgname, "Tests\\%d_%d.jpg", ididx, t + 100);

                    drawOnImage(&(IplImage)outimg, curPose);
                    imwrite(imgname, outimg);
                }

                if (t == sr[t].r)
                    refcurPose = curPose;

                predictpose = sr[t].tree.regress(&image, refcurPose, sr[t].features);
                curPose = curPose.composeWithTarget(predictpose);
            }

            cvSeqPush(seq, &curPose);
        }

	int nComp = cvSeqPartition(seq, 0, &index_seq, PoseParameter::is_equal, 0);

	int *compCount = new int [nComp];
	for (int i = 0; i < nComp; i++) compCount[i] = 0;

	int idx;

	for(int i = 0; i < seq->total; i++)
	{
		idx = *(int*)cvGetSeqElem(index_seq, i);
		compCount[idx] += 1;
	}

	int max_index = 0;

	for (int i = 0; i < nComp; i++)
		if (compCount[max_index] < compCount[i])
			max_index = i;


	predictpose.setzero();
	int count = 0;

	for (int i = 0; i < seq->total; i++)
	{
		idx = *(int*)cvGetSeqElem(index_seq, i);

		if (idx == max_index)
		{
			PoseParameter temp = *(PoseParameter*)cvGetSeqElem(seq, i);

			predictpose = predictpose + temp;
			count++;
		}
	}

    predictpose = predictpose/count;

	cvReleaseMemStorage(&storage);
	cvReleaseMemStorage(&storage_index);

	delete [] initialPose;
	delete [] compCount;
}

    void Write(ofstream &of)
    {
        TreeParam param = sr[0].tree.getParam();
        param.Write(of);

        of << (int)nStage << endl;

        for (int t = 0; t < nStage; t++)
            sr[t].Write(of);
    }

    void Read(ifstream &ifs)
    {
        TreeParam param;
        param.Read(ifs);

        int nstg;
        ifs >> nstg;

        if (nStage > 0) release_cpr(this);
        create_cpr(this, nstg, param);

        for (int t = 0; t < nstg; t++)
        {
            sr[t].Read(ifs);
        }
    }

    static void save_cpr(CPR* cpr, char* cpr_path)
    {
        ofstream of(cpr_path);
        cpr->Write(of);
        of.close();
    }

    static void load_cpr(CPR* cpr, char* cpr_path)
    {
        ifstream ifs(cpr_path);
        cpr->Read(ifs);
        ifs.close();
    }

};

#endif
