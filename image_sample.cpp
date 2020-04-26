#include <iostream>
#include "math.h"
#include "image_sample.hpp"
#include "feature_channel_factory.hpp"

using namespace boost;
using namespace std;
using namespace cv;

ImageSample::ImageSample(const cv::Mat img, std::vector<int> features, FeatureChannelFactory &fcf,
    bool useIntegral_ = false): useIntegral(useIntegral_)
{
    extractFeatureChannels(img, featureChannels, features, useIntegral, fcf);
}

ImageSample::ImageSample(const cv::Mat img, std::vector<int> features, bool useIntegral_ = false):
    useIntegral(useIntegral_)
{
    FeatureChannelFactory fcf = FeatureChannelFactory();
    extractFeatureChannels(img, featureChannels, features, useIntegral, fcf);
}

void ImageSample::extractFeatureChannels(const Mat &img, std::vector<Mat> &vImg, vector<int> features,
    bool useIntegral, FeatureChannelFactory &fcf) const
{
    cv::Mat img_gray;

    if (img.channels() == 1)
    {
        img_gray = img;
    }
    else
    {
        cv::cvtColor(img, img_gray, CV_RGB2GRAY);
    }

    sort(features.begin(), features.end());

    for (unsigned int i = 0; i < features.size(); i++)
    {
        fcf.extractChannel(features[i], useIntegral, img_gray, vImg);
    }
}

void ImageSample::getSubPatches(Rect rect, vector<Mat>& tmpPatches)
{
    for (unsigned int i = 0; i < featureChannels.size(); i++)
        tmpPatches.push_back(featureChannels[i](rect));
}

ImageSample::~ImageSample()
{
    for (unsigned int i = 0; i < featureChannels.size(); i++)
        featureChannels[i].release();

    featureChannels.clear();
}

