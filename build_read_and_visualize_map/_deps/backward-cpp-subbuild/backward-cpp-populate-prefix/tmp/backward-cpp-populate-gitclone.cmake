
if(NOT "/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps/backward-cpp-subbuild/backward-cpp-populate-prefix/src/backward-cpp-populate-stamp/backward-cpp-populate-gitinfo.txt" IS_NEWER_THAN "/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps/backward-cpp-subbuild/backward-cpp-populate-prefix/src/backward-cpp-populate-stamp/backward-cpp-populate-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps/backward-cpp-subbuild/backward-cpp-populate-prefix/src/backward-cpp-populate-stamp/backward-cpp-populate-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E remove_directory "/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps/backward-cpp-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps/backward-cpp-src'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"  clone --no-checkout "https://github.com/bombela/backward-cpp.git" "backward-cpp-src"
    WORKING_DIRECTORY "/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/bombela/backward-cpp.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"  checkout v1.6 --
  WORKING_DIRECTORY "/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps/backward-cpp-src"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'v1.6'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps/backward-cpp-src"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps/backward-cpp-src'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps/backward-cpp-subbuild/backward-cpp-populate-prefix/src/backward-cpp-populate-stamp/backward-cpp-populate-gitinfo.txt"
    "/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps/backward-cpp-subbuild/backward-cpp-populate-prefix/src/backward-cpp-populate-stamp/backward-cpp-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/media/cc/DATA/git/ScanNet-SG/build_read_and_visualize_map/_deps/backward-cpp-subbuild/backward-cpp-populate-prefix/src/backward-cpp-populate-stamp/backward-cpp-populate-gitclone-lastrun.txt'")
endif()

