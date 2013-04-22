#The following command allows the use of the "file" command
cmake_minimum_required(VERSION 2.8)

#The declaration of the project
project(multiboost)

# Multiboost Version
file(READ "version" mbversion)
add_definitions(-DMB_CURRENT_VERSION="${mbversion}")

set(BASEPATH "${CMAKE_SOURCE_DIR}/src")

#This allows recursive parsing of the source files
# Including sub-directories recursively
file(GLOB project_SRCS "${BASEPATH}/*.h")
file(GLOB_RECURSE project_SRCS 
  "${BASEPATH}/Algorithms/*.cpp" "${BASEPATH}/Algorithms/*.h"
  "${BASEPATH}/Bandits/*.cpp" "${BASEPATH}/Bandits/*.h"
  "${BASEPATH}/BanditsLS/*.cpp" "${BASEPATH}/BanditsLS/*.h"
  "${BASEPATH}/Classifiers/*.cpp" "${BASEPATH}/Classifiers/*.h"
  "${BASEPATH}/IO/*.cpp" "${BASEPATH}/IO/*.h"
  "${BASEPATH}/Others/*.cpp" "${BASEPATH}/Others/*.h"
  "${BASEPATH}/StrongLearners/*.cpp" "${BASEPATH}/StrongLearners/*.h"
  "${BASEPATH}/Utils/*.cpp" "${BASEPATH}/Utils/*.h"
  "${BASEPATH}/WeakLearners/*.cpp" "${BASEPATH}/WeakLearners/*.h")

# Bzip2
file(GLOB bzip2_SRCS "${BASEPATH}/Bzip2/*.cpp" "${BASEPATH}/Bzip2/*.c" "${BASEPATH}/Bzip2/*.h")
add_library(Bzip2Lib STATIC ${bzip2_SRCS})

# By Default AlphaReal is double and FeatureReal is float
# uncomment those lines to change that
#ADD_DEFINITIONS( "-DFEATUREREALDOUBLE" )
#ADD_DEFINITIONS( "-DALPHAREALFLOAT" )

#set(CMAKE_INCLUDE_PATH ./)
include_directories("${BASEPATH}")

# make the library (used for unit testing)
add_library(MultiBoostLib STATIC ${project_SRCS})

#This indicates the target (the executable)
add_executable(multiboost "${BASEPATH}/main.cpp")

# adding library to the exec
target_link_libraries(multiboost MultiBoostLib Bzip2Lib)



