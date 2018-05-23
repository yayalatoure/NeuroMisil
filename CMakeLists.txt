cmake_minimum_required(VERSION 3.5)
project(NeuroMisil)

find_package(OpenCV REQUIRED)
find_package(Qt5Widgets REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

file(GLOB SOURCES *.h *.cpp)

set(SOURCE_FILES main.cpp stepdetection.h stepdetection.cpp)
add_executable(NeuroMisil ${SOURCE_FILES} ${SOURCES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(NeuroMisil ${OpenCV_LIBS})

target_link_libraries(NeuroMisil Qt5::Widgets)