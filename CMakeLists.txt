cmake_minimum_required(VERSION 3.8)
project(robot_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Robustly get the major.minor version (e.g., 3.10 instead of 3.1)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sys; print(f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')"
  OUTPUT_VARIABLE PYTHON_INSTALL_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(
  FILES 
    scripts/__init__.py
    scripts/wait_topic.py
    scripts/onnx_session.py
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
)

ament_package()