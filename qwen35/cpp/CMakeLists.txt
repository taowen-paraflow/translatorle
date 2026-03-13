cmake_minimum_required(VERSION 3.20)
project(qwen35_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default OpenVINO location: Python venv inside the project.
# Override with -DOpenVINO_DIR=<path> when configuring.
set(OpenVINO_DIR "C:/Apps/translatorle/.venv/Lib/site-packages/openvino/cmake"
    CACHE PATH "Path to OpenVINO CMake config")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

set(SOURCES
    src/main.cpp
    src/hybrid_model.cpp
    src/hybrid_forward.cpp
    src/hybrid_mtp.cpp
    src/npy_reader.cpp
    src/tokenizer.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    src
    third_party
)

target_link_libraries(${PROJECT_NAME} PRIVATE openvino::runtime)

# MSVC: treat source files as UTF-8.
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /utf-8)
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
