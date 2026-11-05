@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem CPU runtime from the original bundle; no GPU runtime is downloaded or selected.
set "COMPCLASS_MODELS=%~dp0exports\models.json"
set "COMPCLASS_DEMO_DIR=%~dp0demo_stack"
set "ORT_DYLIB_PATH=%~dp0onnxruntime.dll"

rem Never pause CLI commands. A no-argument (double-click) launch always pauses;
rem other GUI launches pause only on failure. Automation can disable both.
set "_cc_pause=error"
if "%~1"=="" set "_cc_pause=always"
for %%C in (classify preprocess bench models --help -h --version -V) do if "%~1"=="%%C" set "_cc_pause=never"
if "%COMPCLASS_NO_PAUSE%"=="1" set "_cc_pause=never"

pushd "%~dp0"
if errorlevel 1 exit /b 1
"%~dp0gpui-demo.exe" %*
set "_cc_exit=%errorlevel%"
popd
if not "%_cc_exit%"=="0" echo Component Classifier exited with code %_cc_exit%. See the output above. 1>&2
if "%_cc_pause%"=="always" pause
if "%_cc_pause%"=="error" if not "%_cc_exit%"=="0" pause
endlocal & exit /b %_cc_exit%
