cmake_minimum_required(VERSION 3.15)

# Explicitly point CMake to your vcpkg installation path
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain")

project(FlightSimulator CXX)

set(CMAKE_CXX_STANDARD 17)

# Find your required packages via vcpkg
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

# Add your executable target matching FlightSimulator.cpp
add_executable(FlightSimulator FlightSimulator.cpp)

# Link libraries using their proper target namespaces (glad::glad and glm::glm)
target_link_libraries(FlightSimulator PRIVATE glfw glad::glad glm::glm)