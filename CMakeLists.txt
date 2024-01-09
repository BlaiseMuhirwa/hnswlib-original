cmake_minimum_required (VERSION 2.6)
project(hnsw_lib
    LANGUAGES CXX)

add_library(hnswlib INTERFACE)
target_include_directories(hnswlib INTERFACE .) 
include(ExternalProject)


function(functionInstallExternalCMakeProject ep_name)
  ExternalProject_Get_Property(${ep_name} binary_dir)
  install(SCRIPT ${binary_dir}/cmake_install.cmake)
endfunction()

ExternalProject_Add(
  ZLIB
  DEPENDS ""
  GIT_REPOSITORY https://github.com/madler/zlib.git
  GIT_TAG v1.2.11
  SOURCE_DIR ZLIB-source
  BINARY_DIR ZLIB-build
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  # INSTALL_COMMAND ""
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX:STRING=${PROJECT_BINARY_DIR}/ep
    -DINSTALL_BIN_DIR:STRING=${PROJECT_BINARY_DIR}/ep/bin
    -DINSTALL_INC_DIR:STRING=${PROJECT_BINARY_DIR}/ep/include
    -DINSTALL_LIB_DIR:STRING=${PROJECT_BINARY_DIR}/ep/lib
    -DINSTALL_MAN_DIR:STRING=${PROJECT_BINARY_DIR}/ep/share/man
    -DINSTALL_PKGCONFIG_DIR:STRING=${PROJECT_BINARY_DIR}/ep/share/pkgconfig
    -DCMAKE_BUILD_TYPE:STRING=Release)
functioninstallexternalcmakeproject(ZLIB)

set(ZLIB_LIB_DEBUG ${PROJECT_BINARY_DIR}/ep/lib/libz.a)
set(ZLIB_LIB_RELEASE ${PROJECT_BINARY_DIR}/ep/lib/libz.a)


ExternalProject_Add(
  CNPY
  DEPENDS ZLIB
  GIT_REPOSITORY https://github.com/sarthakpati/cnpy.git
  # GIT_TAG v1.2.11
  SOURCE_DIR CNPY-source
  BINARY_DIR CNPY-build
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  # INSTALL_COMMAND ""
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS -DZLIB_INCLUDE_DIR:STRING=${PROJECT_BINARY_DIR}/ep/include
             -DZLIB_LIBRARY_DEBUG:STRING=${ZLIB_LIB_DEBUG}
             -DZLIB_LIBRARY_RELEASE:STRING=${ZLIB_LIB_RELEASE}
             -DCMAKE_INSTALL_PREFIX:STRING=${PROJECT_BINARY_DIR}/ep
             -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
             -DCMAKE_BUILD_TYPE:STRING=Release)
functioninstallexternalcmakeproject(CNPY)

include_directories(${PROJECT_BINARY_DIR}/ep/include)

set(CNPY_LIB ${PROJECT_BINARY_DIR}/ep/lib/libcnpy.a)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 11)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      SET( CMAKE_CXX_FLAGS  "-Xclang -Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -ftree-vectorize")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize" )
    endif()

    # examples
    # add_executable(example_search examples/cpp/example_search.cpp)
    # target_link_libraries(example_search hnswlib)

    # add_executable(example_filter examples/cpp/example_filter.cpp)
    # target_link_libraries(example_filter hnswlib)

    # add_executable(example_replace_deleted examples/cpp/example_replace_deleted.cpp)
    # target_link_libraries(example_replace_deleted hnswlib)

    # add_executable(example_mt_search examples/cpp/example_mt_search.cpp)
    # target_link_libraries(example_mt_search hnswlib)

    # add_executable(example_mt_filter examples/cpp/example_mt_filter.cpp)
    # target_link_libraries(example_mt_filter hnswlib)

    # add_executable(example_mt_replace_deleted examples/cpp/example_mt_replace_deleted.cpp)
    # target_link_libraries(example_mt_replace_deleted hnswlib)

    add_executable(dump_graph tests/cpp/dump_graph.cpp)
    target_link_libraries(dump_graph hnswlib ${CNPY_LIB} ${ZLIB_LIB_RELEASE})

    # tests
    # add_executable(test_updates tests/cpp/updates_test.cpp)
    # target_link_libraries(test_updates hnswlib)

    # add_executable(searchKnnCloserFirst_test tests/cpp/searchKnnCloserFirst_test.cpp)
    # target_link_libraries(searchKnnCloserFirst_test hnswlib)

    # add_executable(searchKnnWithFilter_test tests/cpp/searchKnnWithFilter_test.cpp)
    # target_link_libraries(searchKnnWithFilter_test hnswlib)

    # add_executable(multiThreadLoad_test tests/cpp/multiThreadLoad_test.cpp)
    # target_link_libraries(multiThreadLoad_test hnswlib)

    # add_executable(multiThread_replace_test tests/cpp/multiThread_replace_test.cpp)
    # target_link_libraries(multiThread_replace_test hnswlib)

    # add_executable(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    # target_link_libraries(main hnswlib)
endif()
