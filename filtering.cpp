#include "filtering.hpp"

int filtering::HowardFilter::filterMatches(std::vector<Feature3d>& inFeatures, std::vector<Feature3d>& outFeatures)
{
    CV_Assert(inFeatures.size() == outFeatures.size());
    calculateConsistMatrix(inFeatures, outFeatures);

    std::vector<int> indices;
    //initialize clique
    cv::Mat sums(1, consistMatrix.rows, CV_32S);
    for (int row = 0; row < consistMatrix.rows; row++)
        sums.at<int>(0, row) = sum(consistMatrix.row(row))[0];
    cv::Point maxIndex;
    cv::minMaxLoc(sums, 0, 0, 0, &maxIndex, cv::Mat());
    indices.push_back(maxIndex.x);

    int lastAddedIndex = maxIndex.x;

    //initialize compatible matches
    std::vector<int> compatibleMatches, sizes;
    for (int mIndex = 0; mIndex < consistMatrix.cols; mIndex++)
    {
        if (consistMatrix.at<unsigned char>(lastAddedIndex, mIndex) && mIndex != lastAddedIndex)
        {
            compatibleMatches.push_back(mIndex);
            sizes.push_back(sum(consistMatrix.row(mIndex))[0]);
        }
    }

    while(true)
    {
        std::vector<int>::iterator cmIter, sizeIter;
        if (lastAddedIndex != maxIndex.x)
        {
            for (cmIter = compatibleMatches.begin(), sizeIter = sizes.begin();
                 cmIter != compatibleMatches.end(), sizeIter != sizes.end();)
            {
                if (consistMatrix.at<unsigned char>(lastAddedIndex, *cmIter))
                {
                    cmIter++;
                    sizeIter++;
                }
                else
                {
                    cmIter = compatibleMatches.erase(cmIter);
                    sizeIter = sizes.erase(sizeIter);
                }
            }
        }
        if (!compatibleMatches.size())
            break;
        std::vector<int>::iterator maxIter = compatibleMatches.end(), maxSizeIter = sizes.end();
        int maxSize = 0;
        for (cmIter = compatibleMatches.begin(), sizeIter = sizes.begin();
             cmIter != compatibleMatches.end(), sizeIter != sizes.end(); cmIter++, sizeIter++)
        {
            if (*sizeIter > maxSize)
            {
                maxSize = *sizeIter;
                maxSizeIter = sizeIter;
                maxIter = cmIter;
            }
        }
        indices.push_back(*maxIter);
        lastAddedIndex = *maxIter;
        compatibleMatches.erase(maxIter);
        sizes.erase(maxSizeIter);
    }

    std::vector<filtering::Feature3d> filteredInFeatures(indices.size()), filteredOutFeatures(indices.size());
    for (int i = 0; i < indices.size(); i++)
    {
        filteredInFeatures[i] = inFeatures[indices[i]];
        filteredOutFeatures[i] = outFeatures[indices[i]];
    }

    inFeatures = filteredInFeatures;
    outFeatures = filteredOutFeatures;

    return inFeatures.size();
}

void filtering::HowardFilter::calculateConsistMatrix(const std::vector<filtering::Feature3d>& inPoints, const std::vector<filtering::Feature3d>& outPoints)
{
    consistMatrix.create(inPoints.size(), inPoints.size(), CV_8UC1);

    for (int row = 0; row < consistMatrix.rows; row++)
    {
       for (int col = 0; col < row; col++)
       {
           unsigned char consistent = 0;
           cv::Point3f diff1 = inPoints[row].point3d - inPoints[col].point3d;
           cv::Point3f diff2 = outPoints[row].point3d - outPoints[col].point3d;
           if (fabs(cv::norm(diff1) - cv::norm(diff2)) < threshold)
              consistent = 1;
           consistMatrix.at<unsigned char>(row, col) = consistMatrix.at<unsigned char>(col, row) = consistent;
      }
      consistMatrix.at<uint8_t>(row, row) = 1;
    }
}