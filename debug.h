#ifndef DEBUG
#define DEBUG

#include "stdio.h"
#include <cxcore.h>
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <ml.h>
#include <time.h>

void save_CvMat(CvMat* mat, char* file_name);
void load_CvMat(CvMat* mat, char* file_name);
void save_cvRect(CvRect r, char* name);
CvRect load_cvRect(char* name);

void save_IplImage(IplImage* image, char* file_name);
void show_image(IplImage* image, char* window_name);
void show_float_image(IplImage* image, char* window_name);

CvMat* randmat(int m, int n);
CvMat* randperm(int n);
float rand_uniform_debug();

void Substitute(char* pInput, char* pOutput, char* pSrc, char* pDst);

#endif
