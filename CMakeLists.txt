cmake_minimum_required(VERSION 3.15)
project(SLM VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Shared library target
add_library(slm SHARED
    core/src/tokenizer.cpp
    core/src/frequency_store.cpp
    core/src/ngram_model.cpp
    core/src/generator.cpp
    core/src/slm_api.cpp
)

target_include_directories(slm PUBLIC core/include)

# Install
install(TARGETS slm DESTINATION lib)
install(DIRECTORY core/include/slm DESTINATION include)

# Set output directory to python/slm for easy ctypes loading
set_target_properties(slm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/slm"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/slm"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/slm"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/slm"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/slm"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/slm"
)
