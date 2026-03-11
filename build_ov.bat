@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
set PATH=C:\Apps\translatorle\.venv\Lib\site-packages\cmake\data\bin;C:\Apps\translatorle\.venv\Scripts;%PATH%
cd /d C:\Apps\openvino\build
cmake --build . --config Release
if errorlevel 1 (
    echo BUILD_FAILED
    exit /b 1
)
echo BUILD_OK
del C:\Apps\openvino\build\wheels\*.whl 2>nul
cmake --build . --config Release --target ie_wheel
if errorlevel 1 (
    echo WHEEL_FAILED
    exit /b 1
)
echo WHEEL_OK
