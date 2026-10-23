@echo off
rem Build geo4d_gpu: MSVC environment + CMake/Ninja bundled with CLion. Usage: gpu\build.cmd [Release^|Debug]
rem (ASCII only: cmd.exe reads .cmd files in the OEM code page)
setlocal
set CFG=%1
if "%CFG%"=="" set CFG=Release
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set CMAKE=%LOCALAPPDATA%\Programs\CLion\bin\cmake\win\x64\bin\cmake.exe
set NINJA=%LOCALAPPDATA%\Programs\CLion\bin\ninja\win\x64\ninja.exe
"%CMAKE%" -S "%~dp0." -B "%~dp0build" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=%CFG% || exit /b 1
"%CMAKE%" --build "%~dp0build" || exit /b 1
