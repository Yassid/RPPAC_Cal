# CMakeLists.txt for event package. It creates a library with dictionary and a main program
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(analyze_PPAC)


#---Locate the ROOT package and defines a number of variables (e.g. ROOT_INCLUDE_DIRS)
find_package(ROOT REQUIRED COMPONENTS RIO)

#---Define useful ROOT functions and macros (e.g. ROOT_GENERATE_DICTIONARY)
include(${ROOT_USE_FILE})

find_package(OpenCV REQUIRED)
find_package(OpenMP)

if (OPENMP_FOUND)
    MESSAGE("-- OpenMP Support Found")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

endif()

if(OpenCV_FOUND)
	MESSAGE("-- OpenCV Support Found")
endif()

include_directories( ${ROOT_INCLUDE_DIR}
                     ${CMAKE_SOURCE_DIR})

ROOT_GENERATE_DICTIONARY(G__analyze_PPAC analyze_PPAC.h LINKDEF analyze_PPACLinkDef.h)

add_library(analyze_PPAC SHARED analyze_PPAC.C G__analyze_PPAC.cxx)

target_link_libraries(analyze_PPAC ${ROOT_LIBRARIES} ${OpenCV_LIBS})

add_executable(analyze_PPAC_exe analyze_PPAC.C)

target_link_libraries(
analyze_PPAC_exe 
analyze_PPAC
Core
Gpad
Graf
Graf3d
Gui
Hist
Hist
MathCore
Matrix
Minuit
Net
Physics
Postscript
Rint
RIO
Thread
Tree
TreePlayer
${OpenCV_LIBS})


