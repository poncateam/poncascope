cmake_minimum_required(VERSION 3.11)

project("Poncascope")


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
message("\n\n == CMAKE recursively building Ponca\n")
add_subdirectory("external/ponca")

# Add polyscope
message("\n\n == CMAKE recursively building Polyscope\n")
add_subdirectory("external/polyscope")

# Add geometry-central
message("\n\n == CMAKE recursively building geometry-central\n")
add_subdirectory("external/geometry-central")

# Move assets to binary folder
add_custom_target( poncascope-copyassets
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/assets/" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets"
        COMMENT "Install assets to binary folder")

# Create an executable
add_executable( poncascope
                src/main.cpp
                )



add_dependencies( poncascope poncascope-copyassets)

# Link settings
target_link_libraries(poncascope polyscope Ponca::Fitting geometry-central)
