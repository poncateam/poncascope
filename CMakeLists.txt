cmake_minimum_required(VERSION 3.0.0)

project("Poncascope")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Maybe stop from CMAKEing in the wrong place
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Source and build directories cannot be the same. Go use the /build directory.")
endif()

### Configure output locations
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

### Compiler options
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 ) # Emit a compile flags file to support completion engines

# Add ponca
message("\n\n == CMAKE add Ponca\n")
set( PONCA_CONFIGURE_EXAMPLES CACHE BOOL OFF)
set( PONCA_CONFIGURE_TESTS    CACHE BOOL OFF)
set( PONCA_CONFIGURE_DOC      CACHE BOOL OFF)
add_subdirectory("external/ponca")

# Check if Eigen is used from a package or from Ponca submodules
find_package(Eigen3 QUIET)
set(Eigen_Deps "")
if( NOT Eigen3_FOUND ) # Should use Ponca submodule
	message( "Polyscope should use Eigen from ${EIGEN3_INCLUDE_DIRS}" ) 
	include_directories(${EIGEN3_INCLUDE_DIRS})
else()
	set(Eigen_Deps Eigen3::Eigen)
endif()

# Add polyscope
message("\n\n == CMAKE recursively building Polyscope\n")
add_subdirectory("external/polyscope")

# Move assets to binary folder
add_custom_target( poncascope-copyassets
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/assets/" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets"
        COMMENT "Install assets to binary folder")

# Find OpenMP
find_package(OpenMP)
set(OpenMP_link_libraries )
if(OpenMP_CXX_FOUND)
    set(OpenMP_link_libraries OpenMP::OpenMP_CXX)
    message("OpenMP found")
endif()

# Create an executable
add_executable( poncascope
                src/poncaAdapters.hpp
                src/main.cpp
                )

add_dependencies( poncascope poncascope-copyassets)

# Include settings
target_include_directories(poncascope PUBLIC
                            "${CMAKE_CURRENT_SOURCE_DIR}/external/libigl/include"
                            "${CMAKE_CURRENT_SOURCE_DIR}/external/ponca/")

# Link settings
target_link_libraries(poncascope polyscope ${Eigen_Deps} ${OpenMP_link_libraries})


# Fix potential bug on windows (appears with VSCode, but not with VS)
#   Moves bin to project/bin instead of project/bin/BuidType/
set_target_properties(poncascope PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_RUNTIME_OUTPUT_DIRECTORY}>)

# Fix compilation error with MSVC
if (MSVC)
  target_compile_options(poncascope PRIVATE /bigobj)
endif ()

