#ifndef RANDOM_FERN
#define RANDOM_FERN

#include <math.h>
#include <Windows.h>
#include "PIF.h"

#define LOW_THRESHOLD -0.2
#define UP_THRESHOLD 0.2

int cal_fern_ind(double* feature, int* index, double* threshold, int nDepth);
int cal_fern_ind(double* feature, double* threshold, int nDepth);

struct RandomFernParam
{
    int nDepth;
    float beta;
    int nTry;

    void Write(ofstream &of)
    {
        of << nDepth << endl;
        of << beta << endl;
        of << nTry << endl;
    }

    void Read(ifstream &ifs)
    {
        ifs >> nDepth;
        ifs >> beta;
        ifs >> nTry;
    }
};

template<class PoseParameter>
struct RandomFern
{
	int nDepth;
	int* index;
	double* threshold;
	PoseParameter* yFern;

	RandomFern& operator = (const RandomFern &rf)
	{
		nDepth = rf.nDepth;

		for (int i = 0; i < nDepth; i++)
		{
            index[i] = rf.index[i];
            threshold[i] = rf.threshold[i];
		}

		int nBinary = (int)pow((double)2.0, nDepth);

		for (int i = 0; i < nBinary; i++)
            yFern[i] = rf.yFern[i];

		return *this;
	}

	static void create(RandomFern* rf, RandomFernParam param)
    {
        rf->nDepth = param.nDepth;
        rf->index = new int [param.nDepth];
        rf->threshold = new double [param.nDepth];
        rf->yFern = new PoseParameter [(int)pow(2.0f, param.nDepth)];
    }

    static void release(RandomFern* rf)
    {
        delete [] rf->index;
        delete [] rf->threshold;
        delete [] rf->yFern;
    }

    void train_random_fern_regression(double** feature, int nSample, int nFeature, PoseParameter* y,
        int nDepth, float beta, PoseParameter* predict_y, int* plvindices)
    {
        int *sindex = new int [nSample];
        int nBinary = (int)pow(2.0f, nDepth);
        int *count = new int [nBinary];

        for (int i = 0; i < nDepth; i++)
        {
            index[i] = rand()%nFeature;
            threshold[i] = (UP_THRESHOLD - LOW_THRESHOLD)*((double)rand()/RAND_MAX) + LOW_THRESHOLD;
        }

        for (int i = 0; i < nBinary; i++)
        {
            yFern[i].setzero();
            count[i] = 0;
        }

        for (int i = 0; i < nSample; i++)
        {
            sindex[i] = cal_fern_ind(feature[i], index, threshold, nDepth);

            yFern[sindex[i]] = yFern[sindex[i]] + y[i];
            count[sindex[i]]++;
        }

        for (int i = 0; i < nBinary; i++)
        {
            if (count[i] > 0)
            {
                float den = count[i]*(1 + beta/count[i]);
                yFern[i] = yFern[i]/den;
            }
            else
            {
                yFern[i].setzero();
            }
        }

        for (int i = 0; i < nSample; i++)
        {
            predict_y[i] = yFern[sindex[i]];
            plvindices[i] = sindex[i];
        }

        delete [] sindex;
        delete [] count;
    }

    void train(double **f, int nSample, int nFeature, PoseParameter* y,
        RandomFernParam param, int* lvindices, int priority = -1)
    {
        int nVars = PoseParameter::numvars();
        float *weights = new float [nVars];

        calcVWeights(y, nSample, weights);

        PoseParameter *py = new PoseParameter [nSample];
        int *plvindices = new int [nSample];

        RandomFern rf;
        create(&rf, param);

        double loss;
        double lowest_loss = 1000000;

        for (int i = 0; i < param.nTry; i++)
        {
            rf.train_random_fern_regression(f, nSample, nFeature, y, param.nDepth, param.beta, py, plvindices);

            loss = 0;

            for (int j = 0; j < nSample; j++)
                loss = loss + y[j].calcMLoss(py[j], weights, priority);

            loss = sqrt(loss/nSample);

            if (loss < lowest_loss)
            {
                *this = rf;
                for (int j = 0; j < nSample; j++) lvindices[j] = plvindices[j];

                lowest_loss = loss;
            }
        }

        release(&rf);

        delete [] weights;
        delete [] py;
        delete [] plvindices;
    }

    PoseParameter regress(int i)
    {
        return yFern[i];
    }

    template<class FeatureType>
    PoseParameter regress(ImageSample* image, PoseParameter &refcurPose, vector<FeatureType> &features)
    {
        int n = features.size();

        double *f = new double [n];

        for (int i = 0; i < nDepth; i++)
            refcurPose.calcOneFeature(image, features[index[i]], f[index[i]]);

        int idx = cal_fern_ind(f, index, threshold, nDepth);

        delete [] f;

        return yFern[idx];
    }

    RandomFernParam getParam()
    {
        RandomFernParam param;

        param.nDepth = nDepth;
        param.beta = -1;
        param.nTry = -1;

        return param;
    }

    void Write(ofstream &of)
    {
        of << (int)nDepth << endl;

        for (int i = 0; i < nDepth; i++)
        {
            of << (int)index[i] << endl;
            of << (double)threshold[i] << endl;
        }

        int nBinary = (int)pow((double)2.0, nDepth);

        for (int i = 0; i < nBinary; i++)
            yFern[i].Write(of);
    }

    void Read(ifstream &ifs)
    {
        ifs >> nDepth;

        for (int i = 0; i < nDepth; i++)
        {
            ifs >> index[i];
            ifs >> threshold[i];
        }

        int nBinary = (int)pow((double)2.0, nDepth);

        for (int i = 0; i < nBinary; i++)
            yFern[i].Read(ifs);
    }
};

#endif
