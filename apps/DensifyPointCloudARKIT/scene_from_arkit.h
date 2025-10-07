#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>
#include "../../libs/MVS/Common.h"
#include "../../libs/MVS/Scene.h"
#include "../../libs/IO/json.hpp"

namespace MVS::ARKIT {

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
        int index;                  // image index
        bool fixed_width;           // fixed dim when non-uniform resize, `true`: fixed width, `false`: fixed height
        int c;                      // channels
        int width;                  // corresponding to the size of depthmap, confmap, intrinsics but not the image size
        int height;                 // corresponding to the size of depthmap, confmap, intrinsics but not the image size
        std::string image_name;     // image full path
        std::string intrinic_name;  // intrinsic full path
        std::string extrinsic_name; // extrinsic full path
        std::string depth_name;     // depthmap full path
        std::string conf_name;      // confidence map full path

        ARKITFrame() : index(0), fixed_width(true), c(3), width(0), height(0) {}     
    };

    enum class SceneType {
        ARKIT,
        VGGT
    };

    using World2CameraTransformer = std::function<Matrix4x4(const Matrix4x4&)>;

    // `Resize` will introduce some rounding errors, resulting in a slight difference between the original and new aspect ratio.
    // To determine whethe the change in aspect ratio is significant
    // - whethe it is caused by various errors or represents a genuine change, it is necessary calculate the upper bounds of these errors.
    // Based math derivation, if the relative error exceeds 5e-4, the change is considered sigificant.
    // Error formulation: e \approx 0.5/min(w,h), if w,h >1000, e << 0.05%
    inline bool aspectRatioDiff(const cv::Size& src, const cv::Size& target) {
        if(src == target) {
            return false;
        }
        // error threshold
        const double aspect_ratio_error_threshould = 5e-4;

        double e1 = src.width > src.height ? src.aspectRatio(): 1/src.aspectRatio();
        double e2 = src.width > src.height ? target.aspectRatio(): 1/target.aspectRatio();
        
        return std::abs(1 - e2/e1) > aspect_ratio_error_threshould;
    }

    // ARKIT use a right-handed coordinate sysetm with the x-axis pointing to right, 
    // y-axis pointing to up and z-axis pointing to inward(toward to viewer).
    // 
    // However, OpenMVS adopts a different coordinate system: it's y-axis points to downward, 
    // z-axis points to outward - opposite to ARKIT's y and z axes.
    // 
    // Therefore, when coverting the extrinic matrix from ARKIT to OpenMVS, it is necessary to flip
    // the y-axis, z-axis and translation vector to align coordinate systems.     
    inline Matrix4x4 world2Camera_ARKIT(const Matrix4x4& extrinsic) {
        return Matrix4x4::diag(Matrix4x4::diag_type(1,-1,-1, 1)) * extrinsic.inv();
    }
    
    struct ARKITScene {
        // the ownership of `scene` shall belong to OpenMVS codes, 
        // which should handle its construciton and serialization.        
        ARKITScene(Scene *scene, World2CameraTransformer transformer = nullptr, bool toTopLeftCenter = true): scene(scene), toTopLeftCenter(toTopLeftCenter), depthMapSize(), transformer(transformer) {}

        // construct scene instance by type
        static std::unique_ptr<ARKITScene> getInstance(Scene *scene, SceneType type) {
            return std::make_unique<ARKITScene>(scene, type == SceneType::ARKIT ? world2Camera_ARKIT: nullptr);
        }

        // Assembly data structures of OpenMVS.
        // We accept depth map outputs from various models as input, such as ARKit and VGGT. The function will
        // normalize the different inputs into uniform format: intrinsic matrix, image size and camera parameters will be resized to match the size of depth map.
        // 
        // The normalized cameras will be serialized into `scene.mvs`, and OpenMVS will reload this normalized camera and resize it according to the actual image size.
        // 
        // ARKit: The depth map size is a uniform resize of image size, with the intrinsic matrix is corresponding to original image size.
        // VGGT: The depth map size is a non-uniform resize of image size, with the intrinsic is corresponding to the resized image.
        // OpenMVS: Tt only supports uniform resizing of images and intrinsic parameters.
        // 
        // So, we must first calculate the intermediate depth map size using `computeInitDepthMapSize` to convert the non-uniform resizing to uniform one.        
        void build(const std::string& meta_json_path);

        // Return the interpolated depthmap wiht `newSize`
        cv::Mat getDepthMap(const ARKITFrame& frame, const cv::Size& newSize = cv::Size());

        // Typical OpenmMVG + OpenMVS pipeline:
        // 1) OpenMVG generates a sparse point cloud where each point
        // associated with several images, determined through feature matching and bundle adjustment
        // 2) OpenMVS then uses these local releatinships to do the global selection that computing best neighboring views for each image.
        // ARKIT pipeline:
        // 1) ARKIT outputs depthmaps, skiping feature matching and BA
        // 2) Global selection is performed through visibility test using ARKIT depthmap
        void selectViews();

        // Select views for reference image with visibility, 
        // please refers to libs/MVS/SceneDensify.cpp#SelectViews for the detail rules
        void selectViews(Image& imageData, const std::vector<Camera>& cameras, const std::vector<cv::Mat>& depthMaps, float reprojectDepthError = 0.01);

        // The bridge between the OpenMVS pipeline and ARKit, used for depth initialization.
        // In OpenMVS's default implementation, depth maps and normal maps are initialized using 2D Delaunay triangulation.
        // Currently, we need to initialize depth maps using ARKit and compute normal vectors based on pixel and depth relationships.
        void initDepthMap(const ARKITFrame& frame, DepthData& depthData);

        // Projects depth maps generated by ARKIT into a coarse point cloud,
        // witout any point cloud calibrations are performed.
        // This is only used to verify the correctness of coordinate transform from ARKIT to OpenMVS.
        void buildCoarsePointcloud(const std::string& ply_path);

        // Return the duplication of camera
        Camera loadCamera(const ARKITFrame& frame, const cv::Size& newSize = cv::Size(), bool transToRef = true);

        // Load arkit frames from inputs
        void load(const std::string& meta_json_path);

        // initialize scene callback
        void initScene(const std::string& meta_json_path);

        // the function may be called before invoking scene.save, it will clear the resolution of scene.images,
        // Once the resolution becomes invalid, OpenMVS will save the normalized intrinsics and resize them to the actual image size when initializing.
        // The actual image size is deteced by loading only image header. 
        // After then the resized intrinsics will be corresponding perfectly to actual image size.
        // 
        // If the function is not called before scene.save, the current image size (depth map size) and non-normalized intrinsics will be saved,
        // shich will result in a low-resolution reconstruction.
        void clearResolutions();

        // load json
        friend void from_json(const nlohmann::json&, ARKITFrame&);
        
        Scene *scene;

        // arkit frames
        std::vector<ARKITFrame> arkitFrames;

        // raw depthmap size
        cv::Size depthMapSize;

        // convert the intrinsic coordinate system to the center of top-left pixel
        const bool toTopLeftCenter;

        World2CameraTransformer transformer;
    };


}