#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "../../libs/MVS/Common.h"
#include "../../libs/MVS/Scene.h"

namespace MVS::ARKIT {

    static const std::string META_SUFFIX = "_metadata.json";
    static const std::string MASK_SUFFIX = "_mask.png";
    static const std::string DEPTHMAP_SUFFIX = "_depth.raw";
    static const std::string FEATURE_SUFFIX = "_feat.txt";

    // static const double minAngle = FD2R(OPTDENSE::fMinAngle);
    // static const double maxAngle = FD2R(OPTDENSE::fMaxAngle);

    // static const float fOptimAngle = 12.f * M_PI/180;
	// static const float sigmaAngleSmall(-1.f/(2.f*SQUARE(OPTDENSE::fOptimAngle*0.38f)));
	// static const float sigmaAngleLarge(-1.f/(2.f*SQUARE(OPTDENSE::fOptimAngle*0.7f)));

    // reference the score structure from scene.cpp#SelectNeighborViews#Score
   	struct Score {
        Score(): score(0.f), avgScale(0.f), avgAngle(0.f), points(0){}
		float score;
		float avgScale;
		float avgAngle;
		uint32_t points;
        std::vector<Point2f> pixels;
	};

    struct ARKITFrame {
        int index;
        std::string base_name;
        std::string image_name;
        std::string meta_name;
        std::string depthmap_name;
        bool valid;

        ARKITFrame(int index, const std::string& base_name, const std::string& img, const std::string& meta, const std::string& depth)
            : index(index), base_name(base_name), image_name(img), meta_name(meta), depthmap_name(depth), valid(true) {}        
    };

    struct ARKITScene {
        ARKITScene(const std::string& artkit_dir):artkit_dir(artkit_dir), imageSize(), depthMapSize() {}

        // build Scene instance from ARKIT depthmaps
        void build(Scene& scene);

        // Return the interpolated depthmap wiht `newSize`
        cv::Mat getDepthMap(int index, const cv::Size& newSize = cv::Size());

        void selectNeighbors(Scene& scene);
        void selectViews(Image& imageData, const std::vector<Camera>& cameras, const std::vector<cv::Mat>& depthMaps, float reprojectDepthError = 0.01);

        void buildCoarsePointcloud(const Scene& scene, const std::string& ply_path);

        Camera loadCamera(const Scene& scene, int index, cv::Size newSize, bool transToRef = true);

        // arkit dir
        const std::string artkit_dir;

        // arkit frames
        std::vector<ARKITFrame> arkitFrames;

        cv::Size imageSize;

        // raw depthmap size
        cv::Size depthMapSize;

        std::vector<Matrix4x4> extrinsics;
    };

}