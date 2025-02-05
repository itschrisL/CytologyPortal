#include <iostream>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "VLFeatWrapper.cpp"

using namespace std;

namespace segment
{
    class SegmenterTools
    {
    public:
        /*
        runQuickshift takes an image and params and runs Quickshift on it, using the VL_Feat implementation
        Returns:
            cv::Mat = image after quickshift is applied
        Params:
            cv::Mat img = the image
            int kernelsize = the kernel or window size of the quickshift applied
            int maxdist = the largest distance a pixel can be from it's root
        */
        cv::Mat runQuickshift(cv::Mat img, int kernelsize, int maxdist, bool debug = false)
        {
            int channels = img.channels();
            int width = img.cols;
            int height = img.rows;

            cv::Mat tempMat;
            img.copyTo(tempMat);
            tempMat.convertTo(tempMat, CV_64FC3, 1/255.0);
            double* cvimg = (double*) tempMat.data;
            double* vlimg = (double*) calloc(channels*width*height, sizeof(double));

            // create VLFeatWrapper object
            segment::VLFeatWrapper vlf_wrapper = segment::VLFeatWrapper(width, height, channels);
            vlf_wrapper.debug = debug;
            vlf_wrapper.verifyVLFeat();

            // apply quickshift from VLFeat
            vlf_wrapper.convertOPENCV_VLFEAT(cvimg, vlimg);
            int superpixelcount = vlf_wrapper.quickshift(vlimg, kernelsize, maxdist);
            vlf_wrapper.convertVLFEAT_OPENCV(vlimg, cvimg);

            cv::Mat postQuickShift = cv::Mat(height, width, CV_64FC3, cvimg);
            cv::Mat outimg;
            postQuickShift.copyTo(outimg);
            outimg.convertTo(outimg, CV_8UC3, 255);
            free(vlimg);

            if(debug) printf("Super pixels found via quickshift: %i\n", superpixelcount);
            return outimg;
        }

        /*
        runCanny runs canny edge detection on an image, and dilates and erodes it to close holes
        Returns:
            cv::Mat = edges found post dilate/erode
        Params:
            cv::Mat img = image to find edged in
            int threshold1 = first threshold for the hysteresis procedure.
            int threshold2 = second threshold for the hysteresis procedure.
        */
        cv::Mat runCanny(cv::Mat img, int threshold1, int threshold2, bool erodeFlag=false)
        {
            cv::Mat postEdgeDetection;
            img.copyTo(postEdgeDetection);
            cv::Mat blurred;
            cv::blur(img, blurred, cv::Size(2,2));
            cv::Canny(blurred, postEdgeDetection, threshold1, threshold2);

            if(erodeFlag)
            {
                // TODO these values for dilate and erode possibly should be configurable
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2));
                kernel = cv::Mat();
                cv::dilate(postEdgeDetection, postEdgeDetection, kernel, cv::Point(-1, -1), 1);
                cv::erode(postEdgeDetection, postEdgeDetection, kernel, cv::Point(-1, -1), 1);
            }

            return postEdgeDetection;
        }

        /*
        runGmm creates 2 Gaussian Mixture Models, one for cell pixels and one for background pixels,
        then returns the result of the labels generated by these models
        Returns:
            cv::Mat = labels found per pixel
        Params:
            cv::Mat img = image to process
            vector<vector<cv::Point> > hulls = convex hulls to provide initial labeling
            int maxGmmIterations = maximum number of iterations to allow the gmm to train
        */
        cv::Mat runGmm(cv::Mat img, vector<vector<cv::Point> > hulls, int maxGmmIterations)
        {
            int width = img.cols;
            int height = img.rows;

            cv::Mat gray;
            img.convertTo(gray, CV_8UC3);
            cv::cvtColor(gray, gray, CV_BGR2GRAY);
            gray.convertTo(gray, CV_64FC1);

            // create initial probabilities based on convex hulls
            int aSize = width*height;
            float* initialProbs = new float[aSize*2];
            // float* initCellArr = new float[aSize];
            for(int row=0; row < height; row++)
            {
                for(int col=0; col < width; col++)
                {
                    for(unsigned int hullIndex=0; hullIndex < hulls.size(); hullIndex++)
                    {
                        int linearIndex = row*width + col*2;
                        if(cv::pointPolygonTest(hulls[hullIndex], cv::Point2f(col, row), false) >= 0)
                        {
                            initialProbs[linearIndex] = 0;
                            // initCellArr[linearIndex] = 0;
                            initialProbs[linearIndex+1] = 1;
                            break;
                        }
                        else
                        {
                            initialProbs[linearIndex] = 1;
                            // initCellArr[linearIndex] = 1;
                            initialProbs[linearIndex+1] = 0;
                        }
                    }
                }
            }

            gray = gray.reshape(0, gray.rows*gray.cols);
            cv::Mat initialProbMat(aSize, 2, CV_32F, initialProbs);

            // toggle - uncomment to give ALL init probs of one, basically no inital input
            // initialProbMat = cv::Mat::ones(aSize, 2, CV_32F);

            // TODO debugging code, keep?
            // cv::Mat viewInitProbs(aSize, 1, CV_32F, initCellArr);
            // viewInitProbs = viewInitProbs.reshape(0, height*2);
            // viewInitProbs.convertTo(viewInitProbs, CV_8U, 255);
            // cv::imwrite("../images/initialProbs.png", viewInitProbs);

            cv::Mat outputProbs;
            cv::Mat labels;
            cv::Ptr<cv::ml::EM> cell_gmm;
            cv::TermCriteria termCrit = cv::TermCriteria();
            termCrit.type = cv::TermCriteria::COUNT;
            termCrit.maxCount = maxGmmIterations;
            cell_gmm = cv::ml::EM::create();
            cell_gmm->setTermCriteria(termCrit);
            cell_gmm->setClustersNumber(2);
            cell_gmm->trainM(gray, initialProbMat, cv::noArray(), labels, outputProbs);

            labels = labels.reshape(0, img.rows);

            cv::Mat outimg;
            labels.copyTo(outimg);
            outimg.convertTo(outimg, CV_8U, 255);

            delete[] initialProbs;

            return outimg;
        }

        /*
        runMser takes an image and params and runs MSER algorithm on it, for nuclei detection
        Return:
            vector<vector<cv::Point> > = stable regions found
        Params:
            cv::Mat img = the image
            int delta = the # of iterations a region must remain stable
            int minArea = the minimum number of pixels for a viable region
            int maxArea = the maximum number of pixels for a viable region
            double maxVariation = the max amount of variation allowed in regions
            double minDiversity = the min diversity allowed in regions
        */
        vector<vector<cv::Point> > runMser(cv::Mat img, vector<cv::Point> contour, int delta, int minArea, int maxArea,
            double maxVariation, double minDiversity)
        {
            cv::Ptr<cv::MSER> ms = cv::MSER::create(delta, minArea, maxArea, maxVariation, minDiversity);
            cv::Mat tmp;
            img.convertTo(tmp, CV_8U);
            cv::cvtColor(tmp, tmp, CV_BGR2GRAY);
            vector<vector<cv::Point> > regions;
            vector<cv::Rect> mser_bbox;

            ms->detectRegions(tmp, regions, mser_bbox);

            // filter out regions that are outside the clump boundary
            // this is a bit of a hack, but there doesn't seem to be an easy way to make
            // cv::mser run on only a certain region within an image
            int i = 0;
            unsigned int numchecked = 0;
            unsigned int originalsize = regions.size();
            while(regions.size() > 0 && numchecked<originalsize)
            {
                for(cv::Point p : regions[i])
                {
                    if(cv::pointPolygonTest(contour, p, false) < 0)
                    {
                        regions.erase(regions.begin()+i);
                        i--;
                        break;
                    }
                }
                i++;
                numchecked++;
            }

            // TODO add debug check or rm
            if(false) printf("regions found: %lu\n", regions.size());

            return regions;
        }

        /*
        findFinalClumpBoundaries takes an image and a threshold and returns all the contours whose
        size is greater than the threshold
        Returns:
            vector<vector<cv::Point> > = the contours found
        Params:
            cv::Mat img = the input image
            int minAreaThreshold = the minimum area, all contours smaller than this are discarded
        */
        vector<vector<cv::Point> > findFinalClumpBoundaries(cv::Mat img, double minAreaThreshold)
        {
            // opencv wants to find white object on a black background,
            // so we want to invert the labels before findContours
            // update: this was needed, now GMM outputs like this, can probably remove this at some point
            // cv::bitwise_not(img, img);

            vector<vector<cv::Point> > contours;
            cv::findContours(img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
            vector<vector<cv::Point> > clumpBoundaries = vector<vector<cv::Point> >();
            for(unsigned int i=0; i<contours.size(); i++)
            {
                vector<cv::Point> contour = contours[i];
                double area = cv::contourArea(contour);
                if(area > minAreaThreshold)
                {
                    // TODO add debug check or rm
                    if(false) printf("Adding new clump, size:%f threshold:%f\n", area, minAreaThreshold);

                    clumpBoundaries.push_back(contour);
                }
            }

            return clumpBoundaries;
        }

        /*
        extractClump takes an image, contours from the image, and an index, then masks the image to show only
        the contour/clump specified by the index, crops, and returns the image
        Returns:
            cv::Mat = the image of the specified clump - masked out and cropped from the original image
        Params:
            cv::Mat img = the original image
            vector<vector<cv::Point> > clumpBoundaries = the clumpBoundaries in the image
            int clumpIndex = the index in clumpBoundaries of the clump to extract
        */
        cv::Mat extractClump(cv::Mat img, vector<vector<cv::Point> > clumpBoundaries, int clumpIndex)
        {
            // create a mask for each clump and apply it
            cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8U);
            cv::drawContours(mask, clumpBoundaries, clumpIndex, cv::Scalar(255), CV_FILLED);
            cv::Mat fullMasked = cv::Mat(img.rows, img.cols, CV_8U);
            fullMasked.setTo(cv::Scalar(255, 0, 255));
            img.copyTo(fullMasked, mask);
            // invert the mask and then invert the black pixels in the extracted image
            cv::bitwise_not(mask, mask);
            cv::bitwise_not(fullMasked, fullMasked, mask);

            // grab the bounding rect for each clump
            cv::Rect rect = cv::boundingRect(clumpBoundaries[clumpIndex]);

            // create mat of each clump
            cv::Mat clump = cv::Mat(fullMasked, rect);

            clump.convertTo(clump, CV_8UC3);
            return clump;
        }
    };
}
