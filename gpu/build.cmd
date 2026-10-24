@echo off
rem Build geo4d_gpu: MSVC environment + CMake/Ninja bundled with CLion. Usage: gpu\build.cmd [Release^|Debug]
rem Configures the repo root (CMakePresets.json): Release -> gpu\build, Debug -> gpu\build-debug.
rem (ASCII only: cmd.exe reads .cmd files in the OEM code page)
setlocal
set PRESET=release
if /I "%1"=="Debug" set PRESET=debug
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set CMAKE=%LOCALAPPDATA%\Programs\CLion\bin\cmake\win\x64\bin\cmake.exe
set NINJA=%LOCALAPPDATA%\Programs\CLion\bin\ninja\win\x64\ninja.exe
pushd "%~dp0.."
"%CMAKE%" --preset %PRESET% -DCMAKE_MAKE_PROGRAM="%NINJA%" || (popd & exit /b 1)
"%CMAKE%" --build --preset %PRESET% || (popd & exit /b 1)
popd
