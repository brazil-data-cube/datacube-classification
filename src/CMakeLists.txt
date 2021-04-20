cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(smoothing)

include_directories(/usr/local/include)
include_directories("${CMAKE_SOURCE_DIR}/python")
include_directories("${CMAKE_SOURCE_DIR}/python/carma")

file (GLOB SOURCE_FILES "cpp/src/*.cpp")
file (GLOB PYTHON_FILES "python/*.cpp" "python/*.hpp" "python/carma/*.h" "python/carma/carma/*.h")

# Set up such that XCode organizes the files
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES} )

# Armadillo
find_library(ARMADILLO_LIB armadillo HINTS /usr/local/lib/ REQUIRED)

# Pybind11
find_package(pybind11 REQUIRED)
pybind11_add_module(smoothing 
	${SOURCE_FILES}
	${PYTHON_FILES}
)

target_link_libraries(smoothing PUBLIC ${ARMADILLO_LIB})

install(TARGETS smoothing
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
)