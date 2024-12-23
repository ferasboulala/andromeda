cmake_minimum_required(VERSION 3.0)

project(andromeda)

# Set compiler and linker flags for web.
if (USE_EMSCRIPTEN STREQUAL "1")
  set(USE_EMSCRIPTEN TRUE)
else()
  set(USE_EMSCRIPTEN FALSE)
endif()

# Set compiler flags.
if (USE_EMSCRIPTEN)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s USE_GLFW=3 -s ASSERTIONS=1 -s WASM=1 -s ASYNCIFY")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s USE_GLFW=3 -s ASSERTIONS=1 -s WASM=1 -s ASYNCIFY")
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s USE_GLFW=3 -s ASSERTIONS=1 -s WASM=1 -DPLATFORM_WEB")
  add_compile_options(-O3 -ffast-math)
else()
  # Do not use -ffast-math because it will break floating point correctness. Use it only if coordinate types are integers.
  add_compile_options(-O2 -march=native -ffast-math -Wall -Wextra -fverbose-asm -Xclang -fopenmp)
endif()

# Set C++ standard and output directories.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ALL_LIBRARIES)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../bin/")

add_subdirectory(third-party/raylib)

function(create_binary target_name entry_point)
  add_executable(${target_name} ${entry_point})
  target_include_directories(
    ${target_name}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/third-party/spatial-tree/spatial-tree
    ${CMAKE_SOURCE_DIR}/andromeda)
  target_link_libraries(
    ${target_name}
    raylib
    -lomp
    -L/opt/homebrew/opt/libomp/lib
    )
endfunction()

create_binary("demo" examples/demo.cpp)
