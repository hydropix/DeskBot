@echo off
REM Lance DeskBot avec la navigation fluide (Potential Field)
REM Comportement "eau qui contourne" les obstacles

echo.
echo =========================================
echo   DeskBot - Navigation Fluide (Field)
echo =========================================
echo.
echo Controles:
echo   - GUI: Joystick virtuel ou champ Heading
echo   - Clavier: S=arret, ESPACE=pousser, R=reset, N=arret nav, ESC=quitter
echo.
echo Le robot est "repousse" par les obstacles comme par des ressorts.
echo.

.venv\Scripts\python.exe -m deskbot --planner field %*
