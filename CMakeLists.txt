cmake_minimum_required (VERSION 3.9.0)
project (mesh)
include_directories(src)

include_directories(../util/src)
link_directories(../util/build)

# enable IPO
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if( supported )
	message(STATUS "IPO / LTO enabled")
	set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
	message(STATUS "IPO / LTO not supported: <${error}>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -Wno-unused-variable -Wno-register -Werror -fopenmp")

file(GLOB utils "src/util/*.cpp")

set(binaries "su3_sweep" "su3_langevin" "su3_nspt")
foreach(bin ${binaries})
	add_executable(${bin} src/${bin}.cpp ${utils})
	target_link_libraries(${bin} fmt stdc++fs Grid numa z crypto lime fftw3f fftw3 mpfr gmp stdc++ m z numa)

endforeach()
