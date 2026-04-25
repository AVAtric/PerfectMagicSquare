########################################################################################################################
### MAGIC SQUARE APPLICATION
########################################################################################################################

cmake_minimum_required(VERSION 3.22)

########################################################################################################################
### PROJECT DESCRIPTION AND META INFO
########################################################################################################################

set(MAGIC_SQUARE_PROJECT_NAME           "PerfectMagicSquare")
set(MAGIC_SQUARE_PROJECT_DESCRIPTION    "Tool to create solutions for magic squares of size N")

########################################################################################################################
### DECLARE PROJECT
########################################################################################################################

project(${MAGIC_SQUARE_PROJECT_NAME})

enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Suppress deprecation warnings from third-party headers (tabulate.hpp)
add_compile_options(-Wno-deprecated-literal-operator)

INCLUDE(FindPkgConfig)

set(MAGIC_SQUARE_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(MAGIC_SQUARE_INC_DIR ${PROJECT_SOURCE_DIR}/include)
set(MAGIC_SQUARE_TESTS_DIR ${PROJECT_SOURCE_DIR}/tests)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(tests)
