cmake_minimum_required(VERSION 2.8)

project (ocl-basic)

set (CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package( OpenCL )

if(WIN32)
  set (compiler_flags "/WX /EHsc")
else() 
  set (compiler_flags "-Werror -std=c99")
endif(WIN32)

add_executable(${CMAKE_PROJECT_NAME} main.c)

message (STATUS "opencl libs: ${OPENCL_LIBRARIES}")

target_link_libraries(${CMAKE_PROJECT_NAME} ${OPENCL_LIBRARIES})

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES 
                      INCLUDE_DIRECTTORIES ${OPENCL_INCLUDE_DIRS}
                      COMPILE_FLAGS ${compiler_flags})
