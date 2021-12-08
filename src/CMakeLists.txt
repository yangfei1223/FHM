cmake_minimum_required(VERSION 3.6)
project(HFM)

set(CMAKE_CXX_STANDARD 11)

set(EXE_NAME HFM)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)
aux_source_directory(. DIR_SRCS)

add_executable(${EXE_NAME} ${DIR_SRCS})

target_link_libraries(${EXE_NAME} opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)