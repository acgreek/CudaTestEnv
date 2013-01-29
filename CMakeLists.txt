cmake_minimum_required (VERSION 2.8.4)
project (CudaTestEnv)

set (VERSION_MAJOR 1)
set (VERSION_MINOR 0)

ADD_DEFINITIONS(-DCUDA_EMU )

set(PROGS mp1 mp2 mp3 mp4 mp5 mp6)
set(GENAPPS GenDataMP1 GenDataMP2 GenDataMP4 GenDataMP5 GenDataMP6)

set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS thread)

find_package(Threads REQUIRED)
set(TO_BUILD ${PROGS} ${GENAPPS})
set(DEPEND_BT ${PROGS} ) 

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_C_FLAGS "-Wall -Wextra -g3 -O0 " )
set(CMAKE_CXX_FLAGS "-Wall -Wextra -g3 -O0 -std=c++0x" )

if (WIN32 AND __COMPILER_GNU)
   # mingw-gcc fails to link boost::thread
   add_definitions(-DBOOST_THREAD_USE_LIB)
endif (WIN32 AND __COMPILER_GNU)

include_directories ("${PROJECT_SOURCE_DIR}/include")

foreach(_app ${TO_BUILD}) 
	add_executable(${_app} "${_app}.cc")
endforeach()

foreach(_app ${DEPEND_BT}) 
	target_link_libraries(${_app} ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES})
endforeach()

add_custom_command(OUTPUT vecA.txt vecB.txt vecC.txt COMMAND  GenDataMP1 90 DEPENDS GenDataMP1 )
add_custom_command(OUTPUT matA.txt matB.txt matC.txt COMMAND  GenDataMP2 90 10 39 DEPENDS GenDataMP2)
add_custom_command(OUTPUT vecSumA.txt vecSumResult.txt vecSumResult.txt COMMAND GenDataMP4 1000 DEPENDS GenDataMP4)
add_custom_command(OUTPUT vecCumSumA.txt vecCumSumResult.txt vecCumSumResult.txt COMMAND GenDataMP5 100 DEPENDS GenDataMP5)
add_custom_command(OUTPUT imageInput.txt imageOutput.txt COMMAND GenDataMP6 10 10 3 DEPENDS GenDataMP6)

add_custom_target(run1 mp1 vecA.txt vecB.txt vecC.txt DEPENDS vecA.txt vecB.txt vecC.txt )
add_custom_target(run2 mp2 matA.txt matB.txt matC.txt DEPENDS matA.txt matB.txt matC.txt)
add_custom_target(run3 mp3 matA.txt matB.txt matC.txt DEPENDS matA.txt matB.txt matC.txt)
add_custom_target(run4 mp4 vecSumA.txt vecSumResult.txt vecSumResult.txt DEPENDS matA.txt matB.txt matC.txt)
add_custom_target(run5 mp5 vecCumSumA.txt vecCumSumResult.txt vecCumSumResult.txt DEPENDS vecCumSumA.txt vecCumSumResult.txt vecCumSumResult.txt )
add_custom_target(run6 mp6 imageInput.txt imageOutput.txt DEPENDS imageInput.txt imageOutput.txt)

add_dependencies(run1 mp1 vecA.txt)
add_dependencies(run2 mp2 matA.txt matB.txt matC.txt)
add_dependencies(run3 mp3 matA.txt matB.txt matC.txt)
add_dependencies(run4 mp4 vecSumA.txt vecSumResult.txt vecSumResult.txt)
add_dependencies(run5 mp5 vecCumSumA.txt vecCumSumResult.txt vecCumSumResult.txt )
add_dependencies(run6 mp6 imageInput.txt imageOutput.txt)

