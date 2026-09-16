cmake_minimum_required(VERSION 3.21)
project(pentimento LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(pentimento
    src/main.cpp
    src/csv.cpp
)

target_include_directories(pentimento PRIVATE src)

if(MSVC)
    target_compile_options(pentimento PRIVATE /W4 /utf-8)
else()
    target_compile_options(pentimento PRIVATE -Wall -Wextra)
endif()