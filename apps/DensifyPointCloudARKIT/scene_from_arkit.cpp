#include <filesystem>
#include <unordered_map>
// #include "../../libs/Common/Util.inl"
#include "scene_from_arkit.h"

namespace MVS::ARKIT {
    using namespace SEACAVE;
    namespace fs = std::filesystem;
    using json = nlohmann::json;

    // compute normal to the surface given the 4 neighbors
    // Refer: https://arxiv.org/pdf/2304.12031.pdf
    // Codes: libs/MVS/PatchMatchCUDA.cu#ComputeDepthGradient
    static cv::Vec4f computeDepthNormal(const KMatrix& K, const cv::Mat& depthMap, const Point2i& pos) {
        ASSERT(pos.x< depthMap.cols && pos.y <depthMap.rows);

        const cv::Vec4f ZERO = {0, 0, 0, 0};

        if(pos.x <=0 || pos.x >=depthMap.cols -1 || pos.y <=0 || pos.y >= depthMap.rows -1) {
            return ZERO;
        }

        float depth = depthMap.at<float>(pos.y, pos.x);

        if(depth < 1e-6) {
            return ZERO;
        }

        float depth_left = depthMap.at<float>(pos.y, pos.x - 1);
        float depth_right = depthMap.at<float>(pos.y, pos.x + 1);
        float depth_top = depthMap.at<float>(pos.y - 1, pos.x);
        float depth_bottom = depthMap.at<float>(pos.y + 1, pos.x);
        
        if (depth_right < 1e-6 || depth_left < 1e-6 || depth_bottom <1e-6 || depth_top <1e-6) {
            return ZERO;
        }

        float d_u = 0.5f * (depth_right - depth_left);
        float d_v = 0.5f * (depth_bottom - depth_top);

        // compute normal from depth gradient
        Point3f normal(K(0,0) * d_u, K(1,1) *d_v, (K(0,2)- pos.x)*d_u + (K(1,2)- pos.y) * d_v - depth);
        float norm_value = cv::norm(normal);

        if(norm_value < 1e-6) {
            return ZERO;
        }

        normal /= norm_value;

        return {normal.x, normal.y, normal.z, depth};
    }

    // Read depth map from ARKIT
    static cv::Mat readDepthRaw(const std::string& path, int width, int height) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open depth file: " + path);
        }

        cv::Mat depth(height, width, CV_32FC1);
        file.read(reinterpret_cast<char*>(depth.data), width * height * sizeof(float));
        return depth;
    }

    static cv::Mat resizeDepthMap(const cv::Mat& depthMap, const cv::Size& newSize) {
        CV_Assert(depthMap.type() == CV_32FC1);
        CV_Assert(!depthMap.empty());
        // CV_Assert(newSize.width >0 && newSize.height >0 && newSize.width > );
        CV_Assert(cv::countNonZero(depthMap <= 0)==0);


        if(depthMap.size() == newSize) {
            return depthMap;
        }
        
        cv::Mat inverseDepth(1.0f / inverseDepth);
        
        cv::Mat resizedInverseDepth;
        cv::resize(inverseDepth, resizedInverseDepth, newSize, 0, 0, cv::INTER_LINEAR);
        
        return 1.0f / resizedInverseDepth;
    }

    // Compute the depth map size and resize it's aspect ratio to that of original image
    static void computeInitDepthMapSize(const ARKITFrame& frame, cv::Size& initialSize) {
        IMAGEPTR pImage = Image::ReadImageHeader(frame.image_name);

        // load image header
        if(!pImage) {
            throw std::runtime_error("Failed to open image: " + frame.image_name);
        }

        // Initial depth map size is same with the input meta
        initialSize.width = frame.depthmap_width;
        initialSize.height = frame.depthmap_height;

        // Adjust the initial depth map size to match the aspect ratio of original image 
        // 
        // uniform-resize(ARKIT), 
        // - the aspect raido of depth map matches precisely that of image, 
        // - we resize uniformly the image using OpenMVS to match depth map's size.
        // 
        // non-uniform resize(VGGT),
        // - the depth map size is aligned with multiples of 14
        // - re-compute the aspect ratio
        if(aspectRatioDiff(cv::Size(pImage->GetWidth(), pImage->GetHeight()), initialSize)) {
            if(frame.fixed_width) {
                initialSize.height = cv::saturate_cast<int>(pImage->GetHeight() * frame.depthmap_width * 1.0 / pImage->GetWidth());
            } else {
                initialSize.width = cv::saturate_cast<int>(pImage->GetWidth() * frame.depthmap_height * 1.0 / pImage->GetHeight());
            }
        }
    }

    // Load the original image and resize it to match the aspect ratio of depthMapSize
    // The depthMapSize has been resized to same aspect ratio with original image, so image resize is uniformed.
    static void buildImage(const ARKITFrame& frame, Image& image, const cv::Size& depthMapSize) {
        image.ID = frame.index;
        image.poseID = frame.index;
        image.platformID = 0;
        image.cameraID = frame.index;            
        image.name = frame.image_name;

        unsigned resolution = std::max(depthMapSize.width, depthMapSize.height);

        // reload image pixels with corrent size
        if(!image.ReloadImage(resolution)) {
            throw std::runtime_error("Failed to open image: " + frame.image_name);
        }

        ASSERT(image.GetSize() == depthMapSize);
    }

    template <int m, int n>
    static void parseMatrix(const std::string& s, std::vector<double>& vals) {

        if(s.empty()) {
            throw std::runtime_error("Invalidate matrix string: " + s);
        }

        // remove brackets
        std::stringstream ss(s);

        double v;
        char ch;
        while (ss >> v) {
            vals.push_back(v);
            // skip comma
            ss >> ch; 
        }

        if (vals.size() != m * n) {
            throw std::runtime_error("Invalidate matrix string: " + s);
        }
    } 

    // Load intrinsic matrix and rescale it to the size of initial depth map size
    static void buildIntrinsic(const ARKITFrame& frame, KMatrix& kmatrix, const cv::Size& depthMapSize, bool convertIntrinsicSystem) {
        std::ifstream file(frame.intrinic_name);

        if(!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + frame.intrinic_name);
        }

        std::string line;

        if(!std::getline(file, line)) {
            throw std::runtime_error("Invalidate file: " + frame.intrinic_name);
        }

        std::vector<double> vals;
        parseMatrix<3,3>(line, vals);

        file.close();

        KMatrix scale = KMatrix::eye();

        const cv::Size intrinsic_size(frame.image_width, frame.image_height);

        // non-uniform resizing intrinsic
        if(depthMapSize != intrinsic_size) {
            scale(0,0) = scale(1,1) = depthMapSize.width * 1.0/intrinsic_size.width;

            if(aspectRatioDiff(depthMapSize, intrinsic_size)) {
                scale(1,1) = depthMapSize.height * 1.0 / intrinsic_size.height;
            } 
        }

        // rescale intrinsic
        kmatrix = scale * KMatrix(vals.data());

        // convert to OpneMVS image coordinates
        if(convertIntrinsicSystem) {
            kmatrix(0,2) -= 0.5;
            kmatrix(1,2) -= 0.5;
        }
    }

    // ARKIT use a right-handed coordinate sysetm with the x-axis pointing to right, 
    // y-axis pointing to up and z-axis pointing to inward(toward to viewer).
    // 
    // However, OpenMVS adopts a different coordinate system: it's y-axis points to downward, 
    // z-axis points to outward - opposite to ARKIT's y and z axes.
    // 
    // Therefore, when coverting the extrinic matrix from ARKIT to OpenMVS, it is necessary to flip
    // the y-axis, z-axis and translation vector to align coordinate systems. 
    static void buildPose(const ARKITFrame& frame, Platform::Pose& pose, World2CameraTransformer callback = nullptr) {
        std::ifstream file(frame.extrinsic_name);

        std::string line;

        if(!file.is_open() || !std::getline(file, line)) {
            throw std::runtime_error("Failed to open file: " + frame.extrinsic_name);
        }

        std::vector<REAL> vals;
        parseMatrix<4,4>(line, vals);

        Matrix4x4 extrinsic(vals.data());

        // flip y-axis and z-axis
        const Matrix4x4& world2Camera = callback == nullptr? extrinsic: callback(extrinsic);
        
        pose.R = world2Camera.get_minor<3,3>(0,0);
        
        // set the camera center C = -R^{T}t, t= -RC
        pose.C = -pose.R.t() * world2Camera.get_minor<3,1>(0,3);
    }

    static void scoreView(const Point3d& refPoint, float depth, const Camera& refCamra, const Camera& targetCamra, const cv::Mat& targetDepthMap, 
        Score& score, float optimAngle, float sigmaAngleSmall, float sigmaAngleLarge, float reprojectDepthError) {    
        
        const Point2f& targetPixel = targetCamra.ProjectPointP(refPoint);

        // the projection is not inside neighbor camera
        if(!targetCamra.IsInside(targetPixel, Point2f(targetDepthMap.size()))) {
            return;
        }

        double projectDepth = targetCamra.PointDepth(refPoint);
        if(projectDepth < 1e-6) {
            return;
        }

        const Point2i& iPixel(FLOOR2INT(targetPixel));
        double targetDepth = targetDepthMap.at<float>(iPixel.y, iPixel.x);
        
        if(targetDepth < 1e-6) {
            return;
        }

        if(std::abs((projectDepth - targetDepth)/targetDepth) >= reprojectDepthError) {
            return;
        }

        const Point3d& refDirection = refCamra.C - refPoint;
        const Point3d& targetDirection = targetCamra.C -  refPoint;

        // view angle of point
        const float fAngle = ACOS(ComputeAngle(refDirection.ptr(), targetDirection.ptr()));
        
        // angle weight
		const float wAngle = EXP(SQUARE(fAngle- optimAngle)*(fAngle < optimAngle ? sigmaAngleSmall: sigmaAngleLarge));

        // footprint weight
        const float refFootprint = refCamra.GetFootprintImage(depth);
        const float targetFootprint = targetCamra.GetFootprintImage(targetDepth);
        const float fScaleRatio = refFootprint/targetFootprint;

        float wScale = fScaleRatio > 1.6f ? (SQUARE(1.6f/fScaleRatio)): (fScaleRatio < 1.f ? SQUARE(fScaleRatio) : 1.f);
        
        score.pixels.push_back(targetPixel);
        score.points++; 
        score.score += std::max(wAngle, 0.1f) * wScale;
        score.avgScale += fScaleRatio;
        score.avgAngle += fAngle;
    }
    
    void from_json(const json& j, ARKITFrame& info) {
        j.at("index").get_to(info.index);
        j.at("image_name").get_to(info.image_name);
        j.at("c").get_to(info.c);
        j.at("image_width").get_to(info.image_width);
        j.at("image_height").get_to(info.image_height);
        j.at("depthmap_width").get_to(info.depthmap_width);
        j.at("depthmap_height").get_to(info.depthmap_height);
        j.at("depth_name").get_to(info.depth_name);
        j.at("intrinic_name").get_to(info.intrinic_name);
        j.at("extrinsic_name").get_to(info.extrinsic_name);
        j.at("conf_name").get_to(info.conf_name);
    }    
     
    // Typical OpenmMVG + OpenMVS pipeline:
    // 1) OpenMVG generates a sparse point cloud where each point
    // associated with several images, determined through feature matching and bundle adjustment
    // 2) OpenMVS then uses these local releatinships to do the global selection that computing best neighboring views for each image.
    // ARKIT pipeline:
    // 1) ARKIT outputs depthmaps, skiping feature matching and BA
    // 2) Global selection is performed through visibility test using ARKIT depthmap
    void ARKITScene::selectViews() {
        std::vector<Camera> cameras;
        std::vector<cv::Mat> depthMaps;

        for(const ARKITFrame& frame: arkitFrames) {
            const Camera & camera = loadCamera(frame);
            const cv::Mat& depthmap = getDepthMap(frame);

            cameras.push_back(camera);
            depthMaps.push_back(depthmap);
        }

        for(const ARKITFrame& frame: arkitFrames) {
            Image& imageData = scene->images[frame.index];
            selectViews(imageData, cameras, depthMaps);
        }
    }

    // Select views for reference image with visibility, 
    // please refers to libs/MVS/SceneDensify.cpp#SelectViews for the detail rules
    void ARKITScene::selectViews(Image& imageData, const std::vector<Camera>& cameras, const std::vector<cv::Mat>& depthMaps, float reprojectDepthError) {
        const float fMinAngle(FD2R(OPTDENSE::fMinAngle));
        const float fMaxAngle(FD2R(OPTDENSE::fMaxAngle));
        const float fMinArea(OPTDENSE::fMinArea);
        const float fMinScale(0.2f), fMaxScale(3.2f);
        const unsigned nMaxViews(MAXF(OPTDENSE::nMaxViews, OPTDENSE::nNumViews));
        
        const float optimAngle = FD2R(OPTDENSE::fOptimAngle);
        const float sigmaAngleSmall(-1.f/(2.f*SQUARE(optimAngle*0.38f)));
        const float sigmaAngleLarge(-1.f/(2.f*SQUARE(optimAngle*0.7f))); 

        // load camera for depthmap projection
        const Camera& refCamera = cameras[imageData.ID];
        
        std::vector<int> neighbors;
        for(int i = 0; i < arkitFrames.size(); i++) {
            if(i == imageData.ID) {
                continue;
            }
            const Camera& targetCamera = cameras[i];

            // inner product of view angle direction
            auto angleDot = refCamera.Direction().dot(targetCamera.Direction()); 
            double angle = std::acos(angleDot);

            // 3 degree <= angle <= 65 degree
            if(angle >= fMinAngle && angle <= fMaxAngle) {
                neighbors.push_back(i);
            }
        }

        // invalidate the image and frame
        if(neighbors.size() < 2) {
            imageData.poseID = NO_ID;
            return;
        }

        const cv::Mat& depthmap = depthMaps[imageData.ID];

        std::vector<Score> scores(arkitFrames.size());

        // initialize score
        for(int i : neighbors) {
            scores[i].pixels.reserve(depthmap.size().area());
        }

        for(int row = 0; row < depthmap.rows; row++) {
            for(int col = 0; col < depthmap.cols; col++) {
                float depth = depthmap.at<float>(row, col);
                if(depth < 1e-6) {
                    continue;
                }

                const Point3d& refPoint = refCamera.TransformPointI2W(Point3d(col, row, depth));

                for(int j : neighbors) {
                    scoreView(refPoint, depth, refCamera, cameras[j], depthMaps[j], scores[j], 
                        optimAngle,sigmaAngleSmall,sigmaAngleLarge, reprojectDepthError);
                }
                
                // remove points that associated images are less than 3
            }
        }

        for(int i : neighbors) {
            const Score& score = scores[i];
            if(score.points < 3) {
                continue;
            }

            // compute common projection area
            const float area = ComputeCoveredArea<float,2,16,false>((const float *)score.pixels.data(), score.pixels.size(), Point2f(depthMapSize).ptr());
            
            ViewScore& neighbor = imageData.neighbors.AddEmpty();
            neighbor.ID = i;
			neighbor.points = score.points;
			neighbor.scale = score.avgScale/score.points;
			neighbor.angle = score.avgAngle/score.points;
			neighbor.area = area;
			neighbor.score = score.score * MAXF(area, 0.01f); 
        }

        Scene::FilterNeighborViews(imageData.neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, nMaxViews);

        // sort neighbors by score descent
        imageData.neighbors.Sort([](const ViewScore& i, const ViewScore& j) {
			return i.score > j.score;
		});
    }

    // The bridge between the OpenMVS pipeline and ARKit, used for depth initialization.
    // In OpenMVS's default implementation, depth maps and normal maps are initialized using 2D Delaunay triangulation.
    // Currently, we need to initialize depth maps using ARKit and compute normal vectors based on pixel and depth relationships.
    void ARKITScene::initDepthMap(const ARKITFrame& frame, DepthData& depthData) {
        depthData.depthMap.create(depthData.size);
        depthData.normalMap.create(depthData.size);
        
        // load depth map and resize it to corresponding size
        const cv::Mat& depthMap = getDepthMap(frame, depthData.size);

        int invalidDepthmapPixels = 0;

        for(int row = 0; row < depthMap.rows; row++) {
            for(int col = 0; col < depthMap.cols; col++) {
                float depth = depthMap.at<float>(row,col);
                if(depth<1e-6) {
                    invalidDepthmapPixels++;
                }
            }
        }

        // the camera has been resized to corrent size in OpenMVS
        const Camera& camera = depthData.GetCamera();

        float min_depth = std::numeric_limits<float>::max();
        float max_depth = 0.f;

        int invalidDepth =0;
        int invalidNormals =0;
        for(int i = 0; i < depthData.depthMap.rows; i++) {
            for(int j = 0; j < depthData.depthMap.cols; j++) {
                cv::Vec4f v = computeDepthNormal(camera.K, depthMap, {j,i});
                Normal normal(v[0], v[1], v[2]);
                depthData.normalMap(i,j) = normal;
                depthData.depthMap(i,j) = v[3];

                min_depth = std::min(min_depth, v[3]==0.f ? min_depth: v[3]);
                max_depth = std::max(max_depth, v[3]);

                if(v[3] < 1e-6) {
                    invalidDepth++;
                }

                if(cv::norm(normal) <1e-6) {
                    invalidNormals++;
                }                
            }
        }

        depthData.dMin = 0.9 * min_depth;
        depthData.dMax = 1.1 * max_depth;
        VERBOSE("Base DepthMap: Image ID:%d, (%d/%d), invalid depths:%d, invalid normals: %d", 
            frame.index, invalidDepthmapPixels,depthMap.total(), invalidDepth, invalidNormals);
    } 

    // return a duplication of camera
    Camera ARKITScene::loadCamera(const ARKITFrame& frame, const cv::Size& newSize, bool transToRef) {
        cv::Size initSize = newSize.empty() ? depthMapSize : newSize;

        ASSERT(!aspectRatioDiff(depthMapSize, initSize));

        // referenct camera
        const Camera& cameraRef = scene->images[0].camera;
        const Image& image = scene->images[frame.index];

        // resize resolution to `newSize`
        Camera camera = image.camera.GetScaled(image.GetSize(), initSize);

        if(transToRef) {
            auto R = camera.R * cameraRef.R.t();
            auto t = camera.R * (cameraRef.C - camera.C);
            auto C = -R.t() * t;

            camera = Camera(camera.K, R, C);
        }

        return camera;
    }

    // return a duplication of depth map
    cv::Mat ARKITScene::getDepthMap(const ARKITFrame& frame, const cv::Size& newSize) {
        cv::Size initSize = newSize.empty() ? depthMapSize : newSize;
        
        ASSERT(!aspectRatioDiff(depthMapSize, initSize));

        cv::Mat depthmap = readDepthRaw(frame.depth_name, frame.image_width, frame.image_height);

        if (initSize == depthmap.size()) {
            return depthmap;
        }

        // safe depth, set pixel depth to 1e-7 for invalid depths
        depthmap.setTo(1e-7, depthmap <= 0); 

        // resize
        cv::Mat resizedInvDepth;
        cv::resize(1.0f / depthmap, resizedInvDepth, initSize, 0, 0, cv::INTER_LINEAR);

        return 1.0f / resizedInvDepth;
    }

    // Projects depth maps generated by ARKIT into a coarse point cloud,
    // witout any point cloud calibrations are performed.
    // This is only used to verify the correctness of coordinate transform from ARKIT to OpenMVS.    
    void ARKITScene::buildCoarsePointcloud(const std::string& ply_path) {
        ASSERT(!depthMapSize.empty());

        PointCloud pointcloud;
        pointcloud.points.reserve(arkitFrames.size() * depthMapSize.area());

        for(const ARKITFrame& frame: arkitFrames) {
            cv::Mat deptmap = getDepthMap(frame);

            // load camera for depthmap projection
            const Camera& camera = loadCamera(frame);

            const Image& image = scene->images[frame.index];

            ASSERT(depthMapSize == image.GetSize());
            ASSERT(depthMapSize == deptmap.size());

            for(int row = 0; row < deptmap.rows; row++) {
                for(int col = 0; col < deptmap.cols; col++) {
                    float depth = deptmap.at<float>(row, col);
                    if(depth < 1e-6 || std::isnan(depth)) {
                        continue;
                    }
                    
                    // int _row = deptmap.rows -1 - row;
                    // int _col = deptmap.cols -1 - col;

                    const TPoint3<REAL>& P = camera.TransformPointI2W(TPoint3<REAL>(col, row, depth));

                    if(std::isnan(P[0]) || std::isnan(P[1]) || std::isnan(P[2])) {
                        continue;
                    }

                    pointcloud.points.emplace_back(P[0], P[1], P[2]);
                    
                    // BGR to RGB
                    const cv::Vec3b& rgb = image.image.at<cv::Vec3b>(row, col);
                    pointcloud.colors.emplace_back(rgb[2], rgb[1], rgb[0]);
                }
            }
        }

        pointcloud.Save(ply_path);
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
    void ARKITScene::build(const std::string& meta_json_path) {

        // compute the initial depth map size
        load(meta_json_path);

        // only one platform
        Platform& platform = scene->platforms.emplace_back();
        platform.name = "";
        
        // poses
        platform.poses.reserve(arkitFrames.size());
        // init scene images
        scene->images.reserve(arkitFrames.size());

        std::vector<KMatrix> kmatrices;
        kmatrices.reserve(arkitFrames.size());

        for(auto& frame : arkitFrames) {
            // resize the image to the size of depth map
            Image& image = scene->images.emplace_back();
            buildImage(frame, image, depthMapSize);
            
            // resize the intrinsics to the size of depth map
            KMatrix kmatrix;
            buildIntrinsic(frame, kmatrix, depthMapSize, toTopLeftCenter);

            Platform::Pose& pose = platform.poses.emplace_back();
            buildPose(frame, pose, transformer);

            // camera stored in `palatform.camers` must be normalized, them will be serialized to scene.mvs
            Platform::Camera& camera = platform.cameras.emplace_back(kmatrix, RMatrix::IDENTITY, CMatrix::ZERO);

            // normalize intrinic matrix
            // Now, the width and height of intrinsic matrix are same with image size(also depth map size)
            camera.K = camera.GetScaledK(REAL(1)/Camera::GetNormalizationScale(image.width, image.height)); 

            // assemble projection matrix, build image camera
            // Associate image with the intrinsic matrix, the intrinsic will be resize with image size
            image.UpdateCamera(scene->platforms);
        }     
    }

    // Load arkit frames
    void ARKITScene::load(const std::string& meta_json_path) {
        std::ifstream file(meta_json_path);
        if(!file) {
            throw std::runtime_error("Failed to opne file: "+ meta_json_path);
        }

        json j;
        file >> j;
        
        arkitFrames = j.get<std::vector<ARKITFrame>>(); 

        if(arkitFrames.size() < 8) {
            throw std::runtime_error("There are not enough frames: "+ std::to_string(arkitFrames.size()));
        }

        // detect the correct depth map size
        computeInitDepthMapSize(arkitFrames[0], depthMapSize);
    }

    void ARKITScene::initScene(const std::string& meta_json_path) {
        
        load(meta_json_path);

        scene->selecViewsCallback = [&](uint32_t imageID, DepthData& depthData) ->bool {
            depthData.neighbors.CopyOf(scene->images[imageID].neighbors);
            return true;
        };

        scene->initDepthMapCallback = [&](uint32_t imageID, DepthData& depthData){
            return initDepthMap(arkitFrames[imageID], depthData);
        };
    }

    // the function may be called before invoking scene.save, it will clear the resolution of scene.images,
    // Once the resolution becomes invalid, OpenMVS will save the normalized intrinsics and resize them to the actual image size when initializing.
    // The actual image size is deteced by loading only image header. 
    // After then the resized intrinsics will be corresponding perfectly to actual image size.
    // 
    // If the function is not called before scene.save, the current image size (depth map size) and non-normalized intrinsics will be saved,
    // shich will result in a low-resolution reconstruction.
    void ARKITScene::clearResolutions() {
        for(int i=0; i < scene->images.GetSize(); i++) {
            auto& image = scene->images[i];
            
            if(image.IsValid()) {
                image.width = image.height = 0;
            }
        }
    }
}