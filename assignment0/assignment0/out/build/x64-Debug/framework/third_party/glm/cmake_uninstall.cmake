if(NOT EXISTS "D:/RQX/申请材料/申请学校/TUD/Q1/CS4365/assignment0/assignment0/out/build/x64-Debug/install_manifest.txt")
  message(FATAL_ERROR "Cannot find install manifest: D:/RQX/申请材料/申请学校/TUD/Q1/CS4365/assignment0/assignment0/out/build/x64-Debug/install_manifest.txt")
endif()

file(READ "D:/RQX/申请材料/申请学校/TUD/Q1/CS4365/assignment0/assignment0/out/build/x64-Debug/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")
foreach(file ${files})
  message(STATUS "Uninstalling $ENV{DESTDIR}${file}")
  if(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    exec_program(
      "D:/RQX/软件/Visual Studio/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" ARGS "-E remove \"$ENV{DESTDIR}${file}\""
      OUTPUT_VARIABLE rm_out
      RETURN_VALUE rm_retval
      )
    if(NOT "${rm_retval}" STREQUAL 0)
      message(FATAL_ERROR "Problem when removing $ENV{DESTDIR}${file}")
    endif()
  else(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    message(STATUS "File $ENV{DESTDIR}${file} does not exist.")
  endif()
endforeach()
