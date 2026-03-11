/*
* Scene.h
*
* Copyright (c) 2014-2022 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

#ifndef _MVS_SCENE_H_
#define _MVS_SCENE_H_


// I N C L U D E S /////////////////////////////////////////////////

#include <functional>
#include "SceneDensify.h"
#include "Mesh.h"
#include <type_traits>

// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

struct FaceViewInfo {
	using Path = std::filesystem::path;

	// mvg view key
    int mvg_view_key = -1;
	
	// view full name
    std::string view_name;
	
	std::string masked_view;

	std::string colored_model;

	// view depthmap
    std::string depthmap;
	
	std::string vertice_in_face_mask_path;
	std::string mvs_debug_dir;

	bool use_rgb_opt = 0;
	float rgb_weight = 0.1;

	// the num of vertices within face
    int mvg_vertices_in_face = 0;

	int mvg_triangles_count = 0;
	
	// the binary maks to denote whether the vertice is in current view face
	std::vector<unsigned char> vertice_in_face_masks;

	// face parsing label counts
    std::map<std::string, int> label_counts;

	// mvs view key
	int mvs_key = -1;

	std::vector<std::unordered_map<IIndex, Point3f>> faceColors;

	std::set<uint32_t> mvg_triangles;
	std::unordered_set<uint32_t> rastered_triangles;
	std::unordered_map<uint32_t, float> quality_scores;
	std::unordered_map<uint32_t, std::pair<uint32_t, float>> gaussians_scores;
	std::unordered_set<uint32_t> quality_filtered_triangles;

	int outlier_removal_result = 0;

	bool face_in_mask(const Mesh::Face& face) {
		return vertice_in_face_masks[face.x] 
				&& vertice_in_face_masks[face.y] 
				&& vertice_in_face_masks[face.z];
	}

	void rasterFilter(IIndex idxView, TImage<cuint32_t>& faceMap) {
		if(mvs_key == -1 || mvs_key != idxView) {
			return;
		}

		rastered_triangles.clear();

		std::set<Mesh::FIndex> validViews;

		// discard any triangles that are not in facemap
		for (int j=0; j<faceMap.rows; ++j) {
			for (int i=0; i<faceMap.cols; ++i) {
				const Mesh::FIndex& idxFace = faceMap(j,i);
				if (idxFace != NO_ID ) {
					validViews.insert(idxFace);
				}
			}
		}

		std::set_intersection(
			mvg_triangles.begin(), 
			mvg_triangles.end(),
			validViews.begin(), 
			validViews.end(),
			std::inserter(rastered_triangles, rastered_triangles.begin())
		);

		VERBOSE("Image name: %s, (MVG Key: %d, Faces: %d), (MVS Key: %d, MVS Faces(before): %d, MVS Faces(after): %d)", 
			view_name.c_str(), mvg_view_key, mvg_triangles_count,
			mvs_key, mvg_triangles.size(), rastered_triangles.size());
	}

	template <typename T>
	struct is_pair : std::false_type {};

	template <typename K, typename V>
	struct is_pair<std::pair<K, V>> : std::true_type {};

	template <typename T>
	void write_any(std::ofstream& file, const T& val) {
		if constexpr (is_pair<T>::value) {
			write_any(file, val.first);
			write_any(file, val.second);
		} else {
			static_assert(std::is_trivially_copyable_v<T>, "Only POD types allowed for binary write.");
			file.write(reinterpret_cast<const char*>(&val), sizeof(T));
		}
	}	
	template<typename Container>
	void saveContainer(const Container& container, const std::string& suffix, const std::string& desc = "") {
		// base file name
		const std::string filename = std::filesystem::path(view_name).stem().string();

		if (container.empty()) {
			VERBOSE("Image: %s, %s are empty!", filename.c_str(), desc.c_str());
			return;
		}

		if(mvs_debug_dir.empty() || !std::filesystem::exists(mvs_debug_dir)) {
			VERBOSE("Image: %s, mvs output dir is empty!", filename.c_str());
			return;
		}
		
		Path output_path = Path(mvs_debug_dir) / (filename + suffix);

		std::ofstream file(output_path, std::ios::binary);
		if (!file.is_open()) {
			VERBOSE("Failed to open file %s", output_path.string().c_str());
			return;
		}

		uint32_t count = static_cast<uint32_t>(container.size());
		file.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
		
		for (const auto& item : container) {
       		write_any(file, item);
   		}

		file.close();
		VERBOSE("Image: %s, Write %s(%d) to %s", filename.c_str(), desc.c_str(), count, output_path.string().c_str());
	}

	void saveRaster() {
		saveContainer(rastered_triangles, "_raster.bin", "rastered faces");
	}

	void saveQualities() {
		saveContainer(quality_scores, "_quality.bin", "quality faces");
	}

	void saveGaussians() {
		saveContainer(gaussians_scores, "_gaussian.bin", "gaussian faces");
	}

	int validGaussians() {
		return std::count_if(gaussians_scores.begin(), gaussians_scores.end(), [](const auto& pair) { 
			return pair.second.first == 1; 
		});
	}

	void qualityFilter(int remove_bins = 2) {
		if(quality_scores.empty()) {
			return;
		}

		const int num_bins = 20;
		
		std::vector<std::vector<unsigned int>> bin_faces(num_bins);
		// hist
		for (auto const& [face_id, quality] : quality_scores) {
			int bin_idx = quality <= 0.0f ? 0 : static_cast<int>(quality * num_bins);

			if (bin_idx >= num_bins) {
				bin_idx = num_bins - 1;
			}

			bin_faces[bin_idx].push_back(face_id);
		}

		quality_filtered_triangles.clear();
        for (auto const& [face_id, quality] : quality_scores) {
            quality_filtered_triangles.insert(face_id);
        }

		for (int i = 0; i < remove_bins && i < num_bins; ++i) {
			for (unsigned int face_id : bin_faces[i]) {
                quality_filtered_triangles.erase(face_id);
            }
		}
	}
};

// Forward declarations
struct MVS_API DenseDepthMapData;

class MVS_API Scene
{
public:
	PlatformArr platforms; // camera platforms, each containing the mounted cameras and all known poses
	ImageArr images; // images, each referencing a platform's camera pose
	PointCloud pointcloud; // point-cloud (sparse or dense), each containing the point position and the views seeing it
	Mesh mesh; // mesh, represented as vertices and triangles, constructed from the input point-cloud
	OBB3f obb; // region-of-interest represented as oriented bounding box containing the entire scene (optional)
	Matrix4x4 transform; // transformation used to convert from absolute to relative coordinate system (optional)

	unsigned nCalibratedImages; // number of valid images

	unsigned nMaxThreads; // maximum number of threads used to distribute the work load

	std::function<bool(uint32_t, DepthData&)> selecViewsCallback;
	std::function<void(uint32_t, DepthData&)> initDepthMapCallback;
	std::vector<FaceViewInfo> faceViews;
public:
	inline Scene(unsigned _nMaxThreads=0)
		: obb(true), transform(Matrix4x4::IDENTITY), nMaxThreads(Thread::getMaxThreads(_nMaxThreads)) {}

	void Release();
	bool IsValid() const;
	bool IsEmpty() const;
	bool ImagesHaveNeighbors() const;
	bool IsBounded() const { return obb.IsValid(); }
	bool HasTransform() const { return transform != Matrix4x4::IDENTITY; }

	bool LoadInterface(const String& fileName);
	bool SaveInterface(const String& fileName, int version=-1) const;

	bool LoadROI(const String& fileName);
	bool LoadDMAP(const String& fileName);
	bool LoadViewNeighbors(const String& fileName);
	bool SaveViewNeighbors(const String& fileName) const;
	bool Import(const String& fileName);

	enum SCENE_TYPE {
		SCENE_NA = 0,
		SCENE_INTERFACE = 1,
		SCENE_MVS = 2,
		SCENE_IMPORT = 3,
	};
	SCENE_TYPE Load(const String& fileName, bool bImport=false);
	bool Save(const String& fileName, ARCHIVE_TYPE type=ARCHIVE_DEFAULT) const;

	bool EstimatePointCloudNormals(bool bRefine=true);
	bool EstimateSparseSurface(unsigned kNeighbors=16, float sizeScale=0.9f, float normalAngleMax=FD2R(0.f));
	bool EstimateNeighborViewsPointCloud(unsigned maxResolution=16);
	void SampleMeshWithVisibility(REAL sampling=0, unsigned maxResolution=320);
	bool ExportMeshToDepthMaps(const String& baseName);

	bool SelectNeighborViews(uint32_t ID, IndexArr& points, unsigned nMinViews=3, unsigned nMinPointViews=2, float fOptimAngle=FD2R(12), float fWeightPointInsideROI=0.7f);
	void SelectNeighborViews(unsigned nMinViews=3, unsigned nMinPointViews=2, float fOptimAngle=FD2R(12), float fWeightPointInsideROI=0.7f);
	static bool FilterNeighborViews(ViewScoreArr& neighbors, float fMinArea=0.1f, float fMinScale=0.2f, float fMaxScale=2.4f, float fMinAngle=FD2R(3), float fMaxAngle=FD2R(45), unsigned nMaxViews=12);

	bool ExportCamerasMLP(const String& fileName, const String& fileNameScene) const;
	static bool ExportLinesPLY(const String& fileName, const CLISTDEF0IDX(Line3f,uint32_t)& lines, const Pixel8U* colors=NULL, bool bBinary=true);

	// Sub-scene split and save
	struct ImagesChunk {
		std::unordered_set<IIndex> images;
		AABB3f aabb;
	};
	typedef cList<ImagesChunk,const ImagesChunk&,2,16,uint32_t> ImagesChunkArr;
	unsigned Split(ImagesChunkArr& chunks, float maxArea, int depthMapStep=8) const;
	bool ExportChunks(const ImagesChunkArr& chunks, const String& path, ARCHIVE_TYPE type=ARCHIVE_DEFAULT) const;

	// Transform scene
	bool Center(const Point3* pCenter = NULL);
	bool Scale(const REAL* pScale = NULL);
	bool ScaleImages(unsigned nMaxResolution = 0, REAL scale = 0, const String& folderName = String());
	Matrix4x4 ComputeNormalizationTransform(bool bScale = false) const;
	void Transform(const Matrix3x3& rotation, const Point3& translation, REAL scale);
	void Transform(const Matrix3x4& transform);
	bool AlignTo(const Scene&);
	REAL ComputeLeveledVolume(float planeThreshold=0, float sampleMesh=-100000, unsigned upAxis=2, bool verbose=true);
	void AddNoiseCameraPoses(float epsPosition, float epsRotation);
	Scene SubScene(const IIndexArr& idxImages) const;
	Scene& CropToROI(const OBB3f&, unsigned minNumPoints = 3);

	// Estimate and set region-of-interest
	bool EstimateROI(int nEstimateROI=0, float scale=1.f);
	
	// Tower scene
	bool ComputeTowerCylinder(Point2f& centerPoint, float& fRadius, float& fROIRadius, float& zMin, float& zMax, float& minCamZ, const int towerMode);
	void InitTowerScene(const int towerMode);
	size_t DrawCircle(PointCloud& pc,PointCloud::PointArr& outCircle, const Point3f& circleCenter, const float circleRadius, const unsigned nTargetPoints, const float fStartAngle, const float fAngleBetweenPoints);
	PointCloud BuildTowerMesh(const PointCloud& origPointCloud, const Point2f& centerPoint, const float fRadius, const float fROIRadius, const float zMin, const float zMax, const float minCamZ, bool bFixRadius = false);
	
	// Dense reconstruction
	bool DenseReconstruction(int nFusionMode=0, bool bCrop2ROI=true, float fBorderROI=0, float fSampleMeshNeighbors=0);
	bool ComputeDepthMaps(DenseDepthMapData& data);
	void DenseReconstructionEstimate(void*);
	void DenseReconstructionFilter(void*);
	void PointCloudFilter(int thRemove=-1);

	// Mesh reconstruction
	bool ReconstructMesh(float distInsert=2, bool bUseFreeSpaceSupport=true, bool bUseOnlyROI=false, unsigned nItersFixNonManifold=4,
						 float kSigma=2.f, float kQual=1.f, float kb=4.f,
						 float kf=3.f, float kRel=0.1f/*max 0.3*/, float kAbs=1000.f/*min 500*/, float kOutl=400.f/*max 700.f*/,
						 float kInf=(float)(INT_MAX/8));

	// Mesh refinement
	bool RefineMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews, float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize,
		unsigned nMaxFaceArea, unsigned nScales, float fScaleStep, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity, float fGradientStep,
		float fThPlanarVertex=0.f, unsigned nReduceMemory=1);
	#ifdef _USE_CUDA
	bool RefineMeshCUDA(unsigned nResolutionLevel, unsigned nMinResolution, unsigned nMaxViews, float fDecimateMesh, unsigned nCloseHoles, unsigned nEnsureEdgeSize,
		unsigned nMaxFaceArea, unsigned nScales, float fScaleStep, unsigned nAlternatePair, float fRegularityWeight, float fRatioRigidityElasticity, float fGradientStep);
	#endif

	// Mesh texturing
	bool TextureMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned minCommonCameras=0, float fOutlierThreshold=0.f, float fRatioDataSmoothness=0.3f,
		bool bGlobalSeamLeveling=true, bool bLocalSeamLeveling=true, unsigned nTextureSizeMultiple=0, unsigned nRectPackingHeuristic=3, Pixel8U colEmpty=Pixel8U(255,127,39),
		float fSharpnessWeight=0.5f, int ignoreMaskLabel=-1, int maxTextureSize=0, const IIndexArr& views=IIndexArr());

	#ifdef _USE_BOOST
	// implement BOOST serialization
	template <class Archive>
	void serialize(Archive& ar, const unsigned int /*version*/) {
		ar & platforms;
		ar & images;
		ar & pointcloud;
		ar & mesh;
		ar & obb;
		ar & transform;
	}
	#endif
};
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_SCENE_H_
