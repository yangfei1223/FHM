# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.8)
# 设置变量
set(PROJECT_NAME HFM)  #项目名称
project (${PROJECT_NAME})

# 添加子目录
add_subdirectory(src)

