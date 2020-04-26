#include "debug.h"

void save_CvMat(CvMat* mat, char* file_name)
{
	int nRow = mat->rows;
	int nCol = mat->cols;

	FILE* fid = fopen(file_name, "w+");

	for (int i = 0; i < nRow; i++)
	{
		for (int j = 0; j < nCol; j++)
		{
			fprintf(fid, "%f ", (float)cvmGet(mat, i, j));
		}

		fprintf(fid, "\n");
	}

	fclose(fid);
}

void load_CvMat(CvMat* mat, char* file_name)
{
	int nRow = mat->rows;
	int nCol = mat->cols;

	FILE* fid = fopen(file_name, "w+");

	for (int i = 0; i < nRow; i++)
	{
		for (int j = 0; j < nCol; j++)
		{
			fscanf(fid, "%f ", &CV_MAT_ELEM(*mat, float, i, j));
		}

		fscanf(fid, "\n");
	}

	fclose(fid);
}

void save_cvRect(CvRect r, char* name)
{
	FILE* fid = fopen(name, "w+");

	fprintf(fid, "%d\n", r.x);
	fprintf(fid, "%d\n", r.y);
	fprintf(fid, "%d\n", r.width);
	fprintf(fid, "%d\n", r.height);

	fclose(fid);
}

CvRect load_cvRect(char* name)
{
	CvRect r;
	FILE* fid = fopen(name, "r");

	fscanf(fid, "%d\n", &r.x);
	fscanf(fid, "%d\n", &r.y);
	fscanf(fid, "%d\n", &r.width);
	fscanf(fid, "%d\n", &r.height);

	fclose(fid);
	return r;
}

void save_IplImage(IplImage* image, char* file_name)
{
	int nRow = image->height;
	int nCol = image->width;

	FILE* fid = fopen(file_name, "w+");

	for (int i = 0; i < nRow; i++)
	{
		for (int j = 0; j < nCol; j++)
		{
			fprintf(fid, "%f ", (float)*(image->imageData + i*image->widthStep + j));
		}

		fprintf(fid, "\n");
	}

	fclose(fid);
}

void show_image(IplImage* image,char* window_name)
{
	cvNamedWindow(window_name, 1);
	cvShowImage(window_name, image);
}

void show_float_image(IplImage* image, char* window_name)
{
	double min, max;
	cvMinMaxLoc(image, &min, &max);

	IplImage* dst_image = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);// only for one channel float image

	for (int i = 0; i < image->height; i++)
		for (int j = 0; j < image->width; j++)
			CV_IMAGE_ELEM(dst_image, uchar, i, j) = (int)((CV_IMAGE_ELEM(image, float, i, j) - min)/(max - min)*255);

	show_image(dst_image, window_name);
	cvReleaseImage(&dst_image);
}

CvMat* randmat(int m, int n) //generate random matrix 0-1
{
    int larger_int = 10000;
	CvMat* mat = cvCreateMat(m,n,CV_32FC1);

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			CV_MAT_ELEM(*mat, float, i, j) = (float)rand()/RAND_MAX;
		}
	}

	return mat;
}

CvMat* randperm(int n)
{
	CvMat* randvec = randmat(1, n);

	CvMat* index = cvCreateMat(1, n, CV_32SC1);
	cvSort(randvec, NULL, index, CV_SORT_ASCENDING);

	cvReleaseMat(&randvec);
	return index;
}

float rand_uniform_debug()
{
	int n = 10000;
	int m = rand()%n;

	return (float)m/n;
}

void Substitute(char* pInput, char* pOutput, char* pSrc, char* pDst)
{
	char *pi, *po, *p;
	int nSrcLen, nDstLen, nLen;

	pi = pInput;
	po = pOutput;
	nSrcLen = strlen(pSrc);
	nDstLen = strlen(pDst);

	p = strstr(pi, pSrc);

	if (p)
	{
		while (p)
		{
			nLen = (int)(p - pi);
			memcpy(po, pi, nLen);
			memcpy(po + nLen, pDst, nDstLen);
			pi = p + nSrcLen;
			po = po + nLen + nDstLen;
			p = strstr(pi, pSrc);
		}

		strcpy(po, pi);
	}
	else
	{
		strcpy(po, pi);
	}
}

