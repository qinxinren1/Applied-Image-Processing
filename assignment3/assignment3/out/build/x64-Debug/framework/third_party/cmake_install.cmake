# Install script for directory: D:/RQX/申请材料/申请学校/TUD/Q1/AIP/assignment3/assignment3/framework/third_party

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "D:/RQX/申请材料/申请学校/TUD/Q1/AIP/assignment3/assignment3/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("D:/RQX/申请材料/申请学校/TUD/Q1/AIP/assignment3/assignment3/out/build/x64-Debug/framework/third_party/catch2/cmake_install.cmake")
  include("D:/RQX/申请材料/申请学校/TUD/Q1/AIP/assignment3/assignment3/out/build/x64-Debug/framework/third_party/glm/cmake_install.cmake")
  include("D:/RQX/申请材料/申请学校/TUD/Q1/AIP/assignment3/assignment3/out/build/x64-Debug/framework/third_party/fmt/cmake_install.cmake")
  include("D:/RQX/申请材料/申请学校/TUD/Q1/AIP/assignment3/assignment3/out/build/x64-Debug/framework/third_party/stb/cmake_install.cmake")

endif()

