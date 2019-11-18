cmake_minimum_required (VERSION 3.9.0)
project (nspt)

# NOTE: set the compiler flags for the submodules as well

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/bug/opt/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/include -I/bug/opt/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/include -xmic-avx512 -I/bug/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/include -I/home/rid04246/opt/lime/1.3.2/include -fopenmp -O3 -Wall")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# for submodules as well (util needs fmt)
include_directories(/bug/scratch/bus46118/include)

# dependencies should be contained in git submodules
add_subdirectory(util)
add_subdirectory(CLI11)
add_subdirectory(json)

# enable IPO
#include(CheckIPOSupported)
#check_ipo_supported(RESULT supported OUTPUT error)
#if( supported )
#	message(STATUS "IPO / LTO enabled")
#	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
#else()
#	message(STATUS "IPO / LTO not supported: <${error}>")
#endif()

include_directories(/bug/scratch/bus46118/include)
link_directories(/bug/scratch/bus46118/lib)
include_directories(/bug/opt/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/include /bug/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/include /home/rid04246/opt/lime/1.3.2/)

include_directories(src)
include_directories(util/src)
include_directories(json/include)
include_directories(CLI11/include)

find_package(HDF5 REQUIRED)

file(GLOB files_cpp "src/util/*.cpp" "src/qcd/*.cpp" "src/nspt/*.cpp" "src/modules/*.cpp")

add_executable(app src/app.cpp ${files_cpp})
target_include_directories(app PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(app util ${HDF5_LIBRARIES} fmt Grid stdc++ mpicxx mpifort mpi mpigi dl rt pthread mpicxx mpifort mpi mpigi dl rt pthread hdf5_cpp numa z crypto lime mpfr gmp mkl_rt stdc++ m z numa hdf5)

target_compile_definitions(app PRIVATE NSPT_ORDER=${ORDER})
