if(MSVC)
	create_rc_files(DensifyPointCloudARKIT)
	FILE(GLOB LIBRARY_FILES_C "*.cpp" "${CMAKE_CURRENT_BINARY_DIR}/*.rc")
else()
	FILE(GLOB LIBRARY_FILES_C "*.cpp")
endif()

message("Test files: ${LIBRARY_FILES_C}")

FILE(GLOB LIBRARY_FILES_H "*.h" "*.inl")

cxx_executable_with_flags(
		DensifyPointCloudARKIT 
		"Apps" "${cxx_default}" 
		"MVS" 
		DensifyPointCloudARKIT.cpp scene_from_arkit.cpp
		${LIBRARY_FILES_H}
)

cxx_executable_with_flags(
		arkit_tool 
		"Apps" "${cxx_default}" 
		"MVS" 
		test.cpp scene_from_arkit.cpp
		${LIBRARY_FILES_H}
)

# Install
INSTALL(TARGETS DensifyPointCloudARKIT
	EXPORT OpenMVSTargets
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)

	# 新增：安装test可执行文件
# INSTALL(TARGETS arkit_tool
# 	EXPORT OpenMVSTargets
# 	RUNTIME DESTINATION "${INSTALL_BIN_DIR}" COMPONENT bin)