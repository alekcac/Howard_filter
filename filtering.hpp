#pragma once
#include "opencv2/core/core.hpp"

namespace filtering
{
struct Feature3d
{
public:
    Feature3d():point(0.f,0.f), point3d(0.f,0.f,0.f)
    {
    }
    Feature3d(const cv::Point2f& _point, const cv::Point3f& _point3d): point(_point), point3d(_point3d)
    {
    }

    Feature3d& operator=(const Feature3d& rhs)
    {
        if (this != &rhs)
        {
            this->point = rhs.point;
            this->point3d = rhs.point3d;
        }
        return *this;
    }

    Feature3d(const Feature3d& feature)
    {
        point = feature.point;
        point3d = feature.point3d;
    }

    cv::Point2f point;
    cv::Point3f point3d;
};

class MatchFilter
{
public:
    MatchFilter(){}
    virtual int filterMatches(std::vector<Feature3d>& inFeatures, std::vector<Feature3d>& outFeatures)
    {
        CV_Assert(inFeatures.size() == outFeatures.size());
        return inFeatures.size();
    }

    virtual ~MatchFilter(){}
};

class HowardFilter: public MatchFilter
{
public:
    HowardFilter(double _threshold = 0.1f):threshold(_threshold), MatchFilter()
    {
    }

    virtual int filterMatches(std::vector<Feature3d>& inFeatures, std::vector<Feature3d>& outFeatures);
    virtual ~HowardFilter(){}

protected:
    void calculateConsistMatrix(const std::vector<Feature3d>& inPoints, const std::vector<Feature3d>& outPoints);


protected:
    double threshold;
    cv::Mat consistMatrix;
};

};
