cmake_minimum_required(VERSION 3.13)
project(libcamera_opencv_face_detection)

set(CMAKE_CXX_STANDARD 17)

# 找 libcamera
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCAMERA REQUIRED libcamera)

# 找 OpenCV
find_package(OpenCV REQUIRED)

# 編譯參數
add_executable(main main.cpp)

# 加入 include path 與 link flags
target_include_directories(main PRIVATE
    ${LIBCAMERA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(main
    ${LIBCAMERA_LIBRARIES}
    ${OpenCV_LIBS}
)
