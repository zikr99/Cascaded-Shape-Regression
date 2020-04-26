#ifndef CASCADE_POSE_REGRESSION
#define CASCADE_POSE_REGRESSION

#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <vector>
#include <stdio.h>

#include "debug.h"
#include "PIF.h"
#include "tree/tree.hpp"
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
		weights[i] = 1.0f/matvacc.at<double>(0, i);
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
		weights[i] = 1.0f/matvacc.at<double>(0, i);
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

    void cpr_train(vector<string> &images, vector<PoseParameter> &_truePose, vector<PoseParameter> &_startPose,
        int nSample, int nAugment, int nStage, int nStagePart, int nFeature, TreeParam param, int* lbpriorities = NULL)
    {
        PoseParameter *truePose = new PoseParameter [nSample*nAugment];
        PoseParameter *curPose = new PoseParameter [nSample*nAugment];

        for (int i = 0; i < nAugment; i++)
        {
            for (int j = 0; j < nSample; j++)
            {
                truePose[i*nSample + j] = _truePose[j];

                if (i == 0)
                    curPose[i*nSample + j] = _startPose[j];
                else
                    curPose[i*nSample + j] = _startPose[j].disturb();
            }
        }

        //learning loop

        int nAugSample = nAugment*nSample;
        int nVars = PoseParameter::numvars();

        PoseParameter *refcurPose = new PoseParameter [nSample*nAugment];
        PoseParameter *yPose = new PoseParameter [nSample*nAugment];

        for (int i = 0; i < nAugSample; i++)
            yPose[i] = curPose[i].deltaInTarget(truePose[i]);

        float *weights = new float [nVars];
        calcDWeights(yPose, nAugSample, weights);

        char eucname[200];
        sprintf(eucname, "results\\trainingerrors.txt");
        ofstream ecf(eucname);

        double curLoss = 0;
        double eucLoss = 0;

        for (int i = 0; i < nAugSample; i++)
        {
            curLoss = curLoss + curPose[i].calcCLoss(truePose[i], weights, lbpriorities[0]);
            eucLoss = eucLoss + curPose[i].calcEucLoss(truePose[i]);
        }

        curLoss = sqrt(curLoss/nAugSample);
        eucLoss = eucLoss/(nAugSample*(PoseParameter::numvars()/2));

        printf("curLoss: %lf\n", curLoss);
        ecf << eucLoss << endl;

        vector<int> chanfts;
        chanfts.push_back(FC_GRAY);

        vector<FeatureType> positions;

        double** f = new double* [nAugSample];
        for (int i = 0; i < nAugSample; i++) f[i] = new double [nFeature];

        int *lvindices = new int [nAugSample];

        for (int t = 0; t < nStage; t++)
        {
            cout << "Stage: " << t << endl;

            if (t%nStagePart == 0)
            {
                for (int i = 0; i < nAugSample; i++) refcurPose[i] = curPose[i];

                PoseParameter::generate_random_positions(positions, nFeature);

                for (int i = 0; i < nSample; i++)
                {
                    cv::Mat mtimage = cv::imread(images[i], 1);
                    ImageSample spimage(mtimage, chanfts, false);

                    for (int j = 0; j < nAugment; j++)
                        refcurPose[j*nSample + i].calcFeature(&spimage, positions, nFeature, f[j*nSample + i]);

                    if (i%100 == 0) cout << i << " ";
                }

                cout << endl;
            }

            for (int i = 0; i < nAugSample; i++)
                yPose[i] = curPose[i].deltaInTarget(truePose[i]);

            TreeModel<PoseParameter> tree;
            TreeModel<PoseParameter>::create(&tree, param);

            if (lbpriorities)
                tree.train(f, nAugSample, nFeature, yPose, param, lvindices, lbpriorities[t]);
            else
                tree.train(f, nAugSample, nFeature, yPose, param, lvindices);

            //save random fern to cpr

            sr[t].r = t - t%nStagePart;
            sr[t].features = positions;
            sr[t].tree = tree;

            TreeModel<PoseParameter>::release(&tree);

            for (int i = 0; i < nAugSample; i++)
            {
                curPose[i] = curPose[i].composeWithTarget(sr[t].tree.regress(lvindices[i]));
            }

            curLoss = 0;
            eucLoss = 0;

            for (int i = 0; i < nAugSample; i++)
            {
                curLoss = curLoss + curPose[i].calcCLoss(truePose[i], weights, lbpriorities[t]);
                eucLoss = eucLoss + curPose[i].calcEucLoss(truePose[i]);
            }

            curLoss = sqrt(curLoss/nAugSample);
            eucLoss = eucLoss/(nAugSample*(PoseParameter::numvars()/2));

            printf("curLoss: %lf\n", curLoss);
            ecf << eucLoss << endl;

            if (t%50 == 0)
            {
                char intmname[200];
                sprintf(intmname, "results\\model%d.txt", t);

                this->nStage = t + 1;
                save_cpr(this, intmname);
                this->nStage = nStage;

                sprintf(intmname, "results\\curpose%d.txt", t);
                ofstream of(intmname);

                of << (int)nAugSample << endl;
                for (int i = 0; i < nAugSample; i++) curPose[i].Write(of);

                of.close();
            }
        }

        ecf.close();

        ofstream of("lastpose.txt");

        of << (int)nAugSample << endl;
        for (int i = 0; i < nAugSample; i++) curPose[i].Write(of);

        of.close();

        for (int i = 0; i < nAugSample; i++)
            delete [] f[i];

        delete [] f;
        delete [] lvindices;

        delete [] refcurPose;
        delete [] yPose;
        delete [] weights;

        delete [] truePose;
        delete [] curPose;
    }

    void fit(ImageSample &image, PoseParameter &startPose, PoseParameter &predictPose, int nCluster)
    {
        CvMemStorage* storage = cvCreateMemStorage(0);
        CvMemStorage* storage_index = cvCreateMemStorage(0);
        CvSeq *seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(PoseParameter), storage);
        CvSeq *index_seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(int), storage_index);

        PoseParameter* initialPose = new PoseParameter [nCluster];

        initialPose[0] = startPose;

        for (int i = 1; i < nCluster; i++)
            initialPose[i] = startPose.disturb();

        PoseParameter curPose, refcurPose, predictpose;

        for (int i = 0; i < nCluster; i++)
        {
            curPose = initialPose[i];

            for (int t = 0; t < nStage; t++)
            {
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

        predictPose.setzero();
        int count = 0;

        for (int i = 0; i < seq->total; i++)
        {
            idx = *(int*)cvGetSeqElem(index_seq, i);

            if (idx == max_index)
            {
                PoseParameter temp = *(PoseParameter*)cvGetSeqElem(seq, i);

                predictPose = predictPose + temp;
                count++;
            }
        }

        predictPose = predictPose/count;

        cvReleaseMemStorage(&storage);
        cvReleaseMemStorage(&storage_index);

        delete [] initialPose;
        delete [] compCount;
    }

    void fit(string &image, PoseParameter &startPose, PoseParameter &predictPose, int nCluster)
    {   static int iidx = -1; iidx++;
        vector<int> chanfts;
        chanfts.push_back(FC_GRAY);

        cv::Mat mtimage = cv::imread(image, 1);
        ImageSample spimage(mtimage, chanfts, false);

        CvMemStorage* storage = cvCreateMemStorage(0);
        CvMemStorage* storage_index = cvCreateMemStorage(0);
        CvSeq *seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(PoseParameter), storage);
        CvSeq *index_seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(int), storage_index);

        PoseParameter* initialPose = new PoseParameter [nCluster];

        initialPose[0] = startPose;

        for (int i = 1; i < nCluster; i++)
            initialPose[i] = startPose.disturb();

        PoseParameter curPose, refcurPose, predictpose;

        for (int i = 0; i < nCluster; i++)
        {
            curPose = initialPose[i];

            if (i == 0)
            {
                char intfname[200];
                sprintf(intfname, "tests2\\%d_%d.jpg", iidx, 1000);

                cv::Mat outimg = spimage.featureChannels[0].clone();
                drawOnImage(&(IplImage)outimg, curPose);

                imwrite(intfname, outimg);
            }

            for (int t = 0; t < nStage; t++)
            {
                if (t == sr[t].r)
                    refcurPose = curPose;

                predictpose = sr[t].tree.regress(&spimage, refcurPose, sr[t].features);
                curPose = curPose.composeWithTarget(predictpose);

                if (i == 0)
                {
                    char intfname[200];
                    sprintf(intfname, "tests2\\%d_%d.jpg", iidx, t + 1000 + 1);

                    cv::Mat outimg = spimage.featureChannels[0].clone();
                    drawOnImage(&(IplImage)outimg, curPose);

                    imwrite(intfname, outimg);
                }
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

        predictPose.setzero();
        int count = 0;

        for (int i = 0; i < seq->total; i++)
        {
            idx = *(int*)cvGetSeqElem(index_seq, i);

            if (idx == max_index)
            {
                PoseParameter temp = *(PoseParameter*)cvGetSeqElem(seq, i);

                predictPose = predictPose + temp;
                count++;
            }
        }

        predictPose = predictPose/count;

        cvReleaseMemStorage(&storage);
        cvReleaseMemStorage(&storage_index);

        delete [] initialPose;
        delete [] compCount;
    }

    void fit(string &image, PoseParameter &startPose, PoseParameter &predictPose, int nCluster,
        int startStage, int endStage)
    {
        vector<int> chanfts;
        chanfts.push_back(FC_GRAY);

        cv::Mat mtimage = cv::imread(image, 1);
        ImageSample spimage(mtimage, chanfts, false);

        CvMemStorage* storage = cvCreateMemStorage(0);
        CvMemStorage* storage_index = cvCreateMemStorage(0);
        CvSeq *seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(PoseParameter), storage);
        CvSeq *index_seq = cvCreateSeq(0, sizeof(CvSeq), sizeof(int), storage_index);

        PoseParameter* initialPose = new PoseParameter [nCluster];

        initialPose[0] = startPose;

        for (int i = 1; i < nCluster; i++)
            initialPose[i] = startPose.disturb();

        PoseParameter curPose, refcurPose, predictpose;

        for (int i = 0; i < nCluster; i++)
        {
            curPose = initialPose[i];

            for (int t = startStage; t <= endStage; t++)
            {
                if ((t == startStage) || (t == sr[t].r))
                    refcurPose = curPose;

                predictpose = sr[t].tree.regress(&spimage, refcurPose, sr[t].features);
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

        predictPose.setzero();
        int count = 0;

        for (int i = 0; i < seq->total; i++)
        {
            idx = *(int*)cvGetSeqElem(index_seq, i);

            if (idx == max_index)
            {
                PoseParameter temp = *(PoseParameter*)cvGetSeqElem(seq, i);

                predictPose = predictPose + temp;
                count++;
            }
        }

        predictPose = predictPose/count;

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
