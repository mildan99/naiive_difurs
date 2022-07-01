cmake_minimum_required(VERSION 3.0.0)
project(in VERSION 0.1.0)

include(CTest)
enable_testing()

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(fmt REQUIRED )

add_executable(in diffur.cpp)
target_link_libraries(in PUBLIC Eigen3::Eigen fmt::fmt)
target_compile_features(in PUBLIC cxx_std_20)
target_compile_options(in PUBLIC -Wall -Wextra -Wno-volatile -Wshadow -pedantic -Wl,-Map=out.map)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
