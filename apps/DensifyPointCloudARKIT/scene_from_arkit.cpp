#include <filesystem>
#include <unordered_map>
#include "../../libs/IO/json.hpp"
// #include "../../libs/Common/Util.inl"
#include "scene_from_arkit.h"

namespace MVS::ARKIT {
    using namespace SEACAVE;
    namespace fs = std::filesystem;
    using json = nlohmann::json;

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

    static void parseARKITFrames(const std::string& artkit_dir, std::vector<ARKITFrame>& frames) {
        
        if (!fs::exists(artkit_dir) || !fs::is_directory(artkit_dir)) {
            throw std::runtime_error("artkit dir: "+ artkit_dir +" is not exists!");
        }

        frames.clear();

        for(const auto& entry: fs::directory_iterator(artkit_dir)) {
            const auto& path = entry.path();
            
            std::string filename = path.filename().string();
            
            if(filename.ends_with(META_SUFFIX)) {
                const std::string& basename = filename.substr(0,filename.length() - META_SUFFIX.length());
                const fs::path& depth_filename = fs::path(artkit_dir)/(basename + DEPTHMAP_SUFFIX);
                const fs::path& image_filename = fs::path(artkit_dir)/(basename + ".JPG");

                if(!fs::exists(depth_filename)) {
                    throw std::runtime_error("Not found depth file: "+ depth_filename.string());
                }

                if(!fs::exists(image_filename)) {
                    throw std::runtime_error("Not found image file: "+ image_filename.string());
                }

                frames.emplace_back(0, basename, image_filename, fs::absolute(path), depth_filename);
            } 
        }

        // sort images by name acent
        std::sort(frames.begin(), frames.end(), [](auto& frame1, auto& frame2){
            return frame1.base_name < frame2.base_name;
        });

        for(int i=0; i< frames.size(); i++) {
            frames[i].index = i;
        }
    }

    template <int m, int n>
    static void parseMatrix(const std::string& s, std::vector<double>& vals) {

        if(s.empty() || s.front()!='[' || s.back() !=']') {
            throw std::runtime_error("Invalidate matrix string: " + s);
        }

        // remove brackets
        std::stringstream ss(s.substr(1, s.length()-2));

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

    // ARKIT use a right-handed coordinate sysetm with the x-axis pointing to right, 
    // y-axis pointing to up and z-axis pointing to inward(toward to viewer).
    // 
    // However, OpenMVS adopts a different coordinate system: it's y-axis points to downward, 
    // z-axis points to outward - opposite to ARKIT's y and z axes.
    // 
    // Therefore, when coverting the extrinic matrix from ARKIT to OpenMVS, it is necessary to flip
    // the y-axis, z-axis and translation vector to align coordinate systems. 
    static void buildPose(const json& data, const ARKITFrame& entry, Platform::Pose& pose) {
        std::vector<REAL> vals;
        parseMatrix<4,4>(data["transform"], vals);

        // flip y-axis and z-axis
        const Matrix4x4& world2Camera = Matrix4x4::diag(Matrix4x4::diag_type(1,-1,-1, 1)) * Matrix4x4(vals.data()).inv();
        
        pose.R = world2Camera.get_minor<3,3>(0,0);
        
        // set the camera center C = -R^{T}t, t= -RC
        pose.C = -pose.R.t() * world2Camera.get_minor<3,1>(0,3);
    }

    static void buildImage(const json& data, const ARKITFrame& frame, Image& image) {
        image.ID = frame.index;
        image.poseID = frame.index;
        image.platformID = 0;
        image.cameraID = frame.index;            
        image.name = frame.image_name;
        image.width = 1920;//data["exif"]["PixelXDimension"];
        image.height = 1440;//data["exif"]["PixelYDimension"];
        image.scale = 1;

        image.ReloadImage();
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

        for(int i=0; i< arkitFrames.size(); i++) {
            const Camera & camera = loadCamera(i, depthMapSize);
            const cv::Mat& depthmap = getDepthMap(i);

            cameras.push_back(camera);
            depthMaps.push_back(depthmap);
        }

        for(int i=0; i< arkitFrames.size(); i++) {
            Image& imageData = scene->images[i];
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
            scores[i].pixels.reserve(depthMapSize.area());
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

        // sort neighbors by socre descent
        imageData.neighbors.Sort([](const ViewScore& i, const ViewScore& j) {
			return i.score > j.score;
		});
    }

    // The bridge between OpenMVS pipeline and ARKIT used to replace the global selection 
    void ARKITScene::selecViews(DepthData& depthData) {
        const Image* pImageData = depthData.GetView().pImageData;

        if(pImageData == nullptr) {
            throw std::runtime_error("pImageData is nullptr!");
        }

        depthData.neighbors.CopyOf(scene->images[pImageData->cameraID].neighbors);
    }

    Camera ARKITScene::loadCamera(int index, const cv::Size newSize, bool transToRef) {

        if(newSize.empty() || newSize.width <=1 || newSize.height <=1 || std::abs(newSize.aspectRatio() - depthMapSize.aspectRatio())> 1e-6) {
            throw std::runtime_error(String::FormatString("Invalid newSize: (%f, %f)",newSize.width, newSize.height));
        }

        // referenct camera
        const Camera& cameraRef = scene->images[0].camera;
        const Image& image = scene->images[index];

        // resize resolution to `newSize`
        Camera camera = image.camera.GetScaled(cv::Size(image.width, image.height), newSize);

        if(transToRef) {
            auto R = camera.R * cameraRef.R.t();
            auto t = camera.R * (cameraRef.C - camera.C);
            auto C = -R.t() * t;

            camera = Camera(camera.K, R, C);
        }

        return camera;
    }

    void ARKITScene::buildCoarsePointcloud(const std::string& ply_path) {
        PointCloud pointcloud;
        pointcloud.points.reserve(arkitFrames.size() * depthMapSize.area());

        for(int i = 0; i< arkitFrames.size(); i++) {
            auto& frame = arkitFrames[i];
            
            cv::Mat deptmap = getDepthMap(i);

            const Image& image = scene->images[i];

            // load camera for depthmap projection
            const Camera& camera = loadCamera(i, depthMapSize);
            
            cv::Mat scaledImage;

            cv::resize(image.image, scaledImage, depthMapSize, 0, 0, cv::INTER_AREA);

            for(int row = 0; row < deptmap.rows; row++) {
                for(int col = 0; col < deptmap.cols; col++) {
                    float depth = deptmap.at<float>(row, col);
                    if(depth < 1e-6) {
                        continue;
                    }
                    
                    const TPoint3<REAL>& P = camera.TransformPointI2W(TPoint3<REAL>(col, row, depth));
                    pointcloud.points.emplace_back(P[0], P[1], P[2]);
                    
                    // BGR to RGB
                    const cv::Vec3b& rgb = scaledImage.at<cv::Vec3b>(row, col);
                    pointcloud.colors.emplace_back(rgb[2], rgb[1], rgb[0]);
                }
            }
        }

        pointcloud.Save(ply_path);
    }

    void ARKITScene::build(const std::string& artkit_dir) {

        parseARKITFrames(artkit_dir, arkitFrames);

        // only one platform
        Platform& platform = scene->platforms.emplace_back();
        platform.name = "";
        
        // poses
        platform.poses.reserve(arkitFrames.size());
        // init scene images
        scene->images.reserve(arkitFrames.size());

        std::vector<KMatrix> kmatrices;
        kmatrices.reserve(arkitFrames.size());

        std::vector<double> vals;

        for(const auto& entry : arkitFrames) {
            std::ifstream json_file(entry.meta_name);

            const json data = json::parse(json_file);
            
            vals.clear();
            parseMatrix<3,3>(data["intrinsic"], vals);

            Platform::Pose& pose = platform.poses.emplace_back();
            buildPose(data, entry, pose);

            // build image
            Image& image = scene->images.emplace_back();
            buildImage(data, entry, image);

            Platform::Camera& camera = platform.cameras.emplace_back(KMatrix(vals.data()), RMatrix::IDENTITY, CMatrix::ZERO);
            // normalize intrinic matrix
            camera.K = camera.GetScaledK(REAL(1)/Camera::GetNormalizationScale(image.width, image.height)); 

            // assemble projection matrix, build image camera
            image.UpdateCamera(scene->platforms);
        }

        // set raw depthmap size
        depthMapSize.width = 256;
        depthMapSize.height = 192;
    }

    cv::Mat ARKITScene::getDepthMap(int index, const cv::Size& newSize) {
        if (index < 0 || index >= arkitFrames.size()) {
            throw std::runtime_error("Invalid index: " + std::to_string(index));
        }

        cv::Mat depthmap = readDepthRaw(arkitFrames[index].depthmap_name, depthMapSize.width, depthMapSize.height);

        if (newSize.empty() || newSize == depthMapSize) {
            return depthmap;
        }

        if (newSize.area() < depthMapSize.area() || std::abs(newSize.aspectRatio() - depthMapSize.aspectRatio()) > 1e-6) {
            throw std::runtime_error("Invalid depthmap size");
        }

        // safe depth, set pixel depth to 1e-7 for invalid depths
        depthmap.setTo(1e-7, depthmap <= 0); 

        // resize
        cv::Mat resizedInvDepth;
        cv::resize(1.0f / depthmap, resizedInvDepth, newSize, 0, 0, cv::INTER_LINEAR);

        return 1.0f / resizedInvDepth;
    }

    // Serializes only arkit frames
    void ARKITScene::save(const std::string& meta_json_path) {
        json arkitScene;
        arkitScene["frames"] = json::array();
        
        for(auto& f : arkitFrames) {
            arkitScene["frames"].push_back({
                {"index", f.index},
                {"base_name", f.base_name},
                {"image_name", f.image_name},
                {"meta_name", f.meta_name},
                {"depthmap_name", f.depthmap_name}
            });
        }

        arkitScene["depthmapWidth"] = depthMapSize.width;
        arkitScene["depthmapHeight"] = depthMapSize.height;
        
        std::ofstream file(meta_json_path);

        if(!file) {
            throw std::runtime_error("Failed to opne file: "+ meta_json_path);
        }

        file << arkitScene.dump(4);

        std::cout << "Save frames to " << meta_json_path << std::endl;
    }

    // Load arkit frames
    void ARKITScene::load(const std::string& meta_json_path) {
        std::ifstream file(meta_json_path);

        if(!file) {
            throw std::runtime_error("Failed to opne file: "+ meta_json_path);
        }

        json j;
        file >> j;
        
        int depthmapWidth = j["depthmapWidth"].get<int>();
        int depthmapHeight = j["depthmapHeight"].get<int>();

        depthMapSize = cv::Size(depthmapWidth, depthmapHeight);

        for (auto& frame : j["frames"]) {
            arkitFrames.emplace_back(
                frame["index"].get<int>(),
                frame["base_name"].get<std::string>(),
                frame["image_name"].get<std::string>(),
                frame["meta_name"].get<std::string>(),
                frame["depthmap_name"].get<std::string>()
            );
        }
    }
}