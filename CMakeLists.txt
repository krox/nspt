cmake_minimum_required (VERSION 3.9.0)
project (nspt)

# NOTE: set the compiler flags for the submodules as well

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Werror -pedantic -fopenmp")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# dependencies should be contained in git submodules
add_subdirectory(util)
add_subdirectory(CLI11)
add_subdirectory(json)

# enable IPO
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if( supported )
	message(STATUS "IPO / LTO enabled")
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
	message(STATUS "IPO / LTO not supported: <${error}>")
endif()

include_directories(src)
include_directories(util/src)
include_directories(json/include)
include_directories(CLI11/include)

find_package(HDF5 REQUIRED)

file(GLOB utils "src/util/*.cpp" "src/nspt/*.cpp" "src/modules/*.cpp")

set(binaries "su3_sweep" "su3_langevin" "su3_nspt" "app")
foreach(bin ${binaries})
	add_executable(${bin} src/${bin}.cpp ${utils})
	target_include_directories(${bin} PRIVATE ${HDF5_INCLUDE_DIRS})
	target_link_libraries(${bin} util ${HDF5_LIBRARIES} fmt stdc++fs Grid numa z crypto lime fftw3f fftw3 mpfr gmp stdc++ m z numa)
	target_compile_definitions(${bin} PRIVATE NSPT_ORDER=${ORDER})
endforeach()
