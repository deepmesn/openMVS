#pragma once
#include "scene_from_arkit.h"

namespace MVS::VGGT {
    using namespace MVS::ARKIT;

    struct VGGTScene: public ARKITScene {
        // build Scene instance from ARKIT depthmaps,
        void build(const std::string& vggt_meta_path);        
    };
}