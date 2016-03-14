@echo off
IF EXIST "devel_env.bat" GOTO ok
echo Use this launcher only when a devel_env.bat file has been created
pause
goto done
:ok
call devel_env.bat
IF EXIST "%VS120COMNTOOLS%\..\IDE\WDExpress.exe" GOTO vcexpress12
IF EXIST "%VS100COMNTOOLS%\..\IDE\devenv.exe" GOTO visual10
IF EXIST "%VS100COMNTOOLS%\..\IDE\vcexpress.exe" GOTO vcexpress10
IF EXIST "%VS90COMNTOOLS%\..\IDE\devenv.exe" GOTO visual9
IF EXIST "%VS90COMNTOOLS%\..\IDE\vcexpress.exe" GOTO vcexpress9
IF EXIST "%VS80COMNTOOLS%\..\IDE\devenv.exe" GOTO visual8
IF EXIST "%VS80COMNTOOLS%\..\IDE\vcexpress.exe" GOTO vcexpress8
echo Cannot find visual
pause
goto done
:vcexpress8
"%VS80COMNTOOLS%\..\IDE\vcexpress" .sln
goto done
:visual8
"%VS80COMNTOOLS%\..\IDE\devenv" .sln
goto done
:vcexpress9
"%VS90COMNTOOLS%\..\IDE\vcexpress" .sln
goto done
:visual9
"%VS90COMNTOOLS%\..\IDE\devenv" .sln
goto done
:vcexpress10
"%VS100COMNTOOLS%\..\IDE\vcexpress" .sln
goto done
:visual10
"%VS100COMNTOOLS%\..\IDE\devenv" .sln
goto done
:visual12
"%VS120COMNTOOLS%\..\IDE\WDExpress" .sln
goto done
:done
	