cmake_minimum_required(VERSION 3.8)
project(hexapod_bringup)

find_package(ament_cmake REQUIRED)

# When the first launch or parameter file is added, install its directory:
# install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
