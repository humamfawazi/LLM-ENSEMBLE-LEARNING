@echo off
rem Jalankan 9 run R3_02 (IndoBERT -> XLM-R -> mBERT) dengan Python pelatihan. Klik dua kali file ini.
rem Bila berhenti karena apa pun, klik dua kali lagi: run yang selesai dilewati, run yang terputus dilanjutkan.
rem Argumen opsional diteruskan, mis.: jalankan_R3_02_semua.bat indobert
title R3_02 - latih ulang 9 run (JANGAN tutup jendela ini)
cd /d "%~dp0"
python R3_02_run_all.py %*
echo.
echo Selesai / berhenti dengan kode %ERRORLEVEL%. Lihat outputs_round3\logs\run_all.log
pause
