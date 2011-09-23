#The following command allows the use of the "file" command
cmake_minimum_required(VERSION 2.6)

#The declaration of the project
project(multiboost)

set(BASEPATH "${CMAKE_SOURCE_DIR}/src")

#This allows recursive parsing of the source files
# Including sub-directories recursively
file(GLOB_RECURSE project_SRCS "${BASEPATH}/*" )

# By Default AlphaReal is double and FeatureReal is float
# uncomment those lines to change that
#ADD_DEFINITIONS( "-DFEATUREREALDOUBLE" )
#ADD_DEFINITIONS( "-DALPHAREALFLOAT" )

#set(CMAKE_INCLUDE_PATH ./)
include_directories("${BASEPATH}")

#This indicates the target (the executable)
add_executable(multiboost ${project_SRCS})

