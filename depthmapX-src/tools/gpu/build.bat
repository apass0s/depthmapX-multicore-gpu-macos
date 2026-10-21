@echo off
rem Builds depthmapXcli, depthmapX (GUI) and depthmapX_gpu.dll with MSVC + Ninja.
rem
rem   tools\gpu\build.bat [build dir] [Qt prefix]
rem
rem Defaults: build dir C:\dmx\build (keep it short: MSVC fails on long paths, and keep
rem it outside OneDrive), Qt C:\Qt\5.15.2\msvc2019_64. Needs Visual Studio 2022 (Build
rem Tools are enough), CMake, Ninja and, for the GPU module, the CUDA Toolkit
rem (found through PATH or CUDA_PATH).
setlocal
set "SRC=%~dp0..\.."
set "BUILD=%~1"
if "%BUILD%"=="" set "BUILD=C:\dmx\build"
set "QTDIR=%~2"
if "%QTDIR%"=="" set "QTDIR=C:\Qt\5.15.2\msvc2019_64"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (echo vswhere.exe not found: install Visual Studio 2022 Build Tools & exit /b 1)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if "%VSDIR%"=="" (echo No Visual Studio installation with the C++ tools found & exit /b 1)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"

cmake -S "%SRC%" -B "%BUILD%" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_PREFIX_PATH=%QTDIR%" || exit /b 1
cmake --build "%BUILD%" --target depthmapXcli depthmapX || exit /b 1
echo Built: %BUILD%\depthmapXcli\depthmapXcli.exe and %BUILD%\depthmapX\depthmapX.exe
