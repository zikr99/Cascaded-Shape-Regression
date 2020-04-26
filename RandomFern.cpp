#include "RandomFern.h"

int cal_fern_ind(double* feature, int* index, double* threshold, int nDepth)
{
	int ind = 0;

	for (int i = 0; i < nDepth; i++)
	{
		ind = 2*ind;
		if (feature[index[i]] < threshold[i]) ind += 1;
	}

	return ind;
}

int cal_fern_ind(double* feature, double* threshold, int nDepth)
{
	int ind = 0;

	for (int i = 0; i < nDepth; i++)
	{
		ind = 2*ind;
		if (feature[i] < threshold[i]) ind += 1;
	}

	return ind;
}
