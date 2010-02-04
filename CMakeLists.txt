#The following command allows the use of the "file" command
cmake_minimum_required(VERSION 2.6)

#The declaration of the project
project(multiboost)

set(BASEPATH "${CMAKE_SOURCE_DIR}/src")

#This allows recursive parsing of the source files
# Including sub-directories recursively
FILE(GLOB_RECURSE project_SRCS "${BASEPATH}/*" )


include_directories("${PROJECT_BINARY_DIR}/src")

list(REMOVE_ITEM project_SRCS "./build/*" )

#message(STATUS "Just to check" ${source_files})

# By Default AlphaReal is double and FeatureReal is float
# uncomment those lines to change that
#ADD_DEFINITIONS( "-DFEATUREREALDOUBLE" )
#ADD_DEFINITIONS( "-DALPHAREALFLOAT" )

#set(CMAKE_INCLUDE_PATH ./)
INCLUDE_DIRECTORIES("${BASEPATH}")

#This indicates the target (the executable)
ADD_EXECUTABLE(multiboost ${project_SRCS})

