cmake_minimum_required(VERSION 3.21)
project(pentimento LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORT_DIR ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)

add_executable(pentimento
    src/main.cpp
    src/csv.cpp
    src/image.cpp
    src/clip.cpp 
    src/embed.cpp)

target_include_directories(pentimento PRIVATE src)
target_include_directories(pentimento SYSTEM PRIVATE
    third_party/stb
    ${ORT_DIR}/include
)

target_link_libraries(pentimento PRIVATE ${ORT_DIR}/lib/onnxruntime.lib)


if(MSVC)
    target_compile_options(pentimento PRIVATE /W4 /utf-8)
else()
    target_compile_options(pentimento PRIVATE -Wall -Wextra)
endif()

add_custom_command(TARGET pentimento POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ORT_DIR}/lib/onnxruntime.dll
            $<TARGET_FILE_DIR:pentimento>)