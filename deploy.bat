@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

for /f %%A in ('echo prompt $E^| cmd') do set "ESC=%%A"

set "GREEN=%ESC%[92m"
set "RED=%ESC%[91m"
set "YELLOW=%ESC%[93m"
set "BLUE=%ESC%[94m"
set "CYAN=%ESC%[96m"
set "BOLD=%ESC%[1m"
set "NC=%ESC%[0m"

set "LOGFILE=deploy.log"

goto :main

:print_header
echo.
echo %CYAN%╔══════════════════════════════════════════════════════╗%NC%
echo %CYAN%║%NC%  %BOLD%ACE-Step Music Generation API — Docker Deploy%NC%       %CYAN%║%NC%
echo %CYAN%╚══════════════════════════════════════════════════════╝%NC%
echo.
exit /b

:print_status
echo %GREEN%[✓]%NC% %~1
exit /b

:print_warn
echo %YELLOW%[!]%NC% %~1
exit /b

:print_error
echo %RED%[✗]%NC% %~1
exit /b

:print_info
echo %BLUE%[→]%NC% %~1
exit /b

:log_init
echo === ACE-Step Deploy Log === > "!LOGFILE!"
echo Дата: %date% %time% >> "!LOGFILE!"
echo Режим: %~1 >> "!LOGFILE!"
echo =========================== >> "!LOGFILE!"
echo. >> "!LOGFILE!"
exit /b

:run_docker
set "RD_DESC=%~1"
set "RD_CMD=%~2"
call :print_info "!RD_DESC!"
echo. >> "!LOGFILE!"
echo ^>^>^> !RD_DESC! >> "!LOGFILE!"
echo ^>^>^> Команда: !RD_CMD! >> "!LOGFILE!"
echo. >> "!LOGFILE!"
set "RD_TMPLOG=%TEMP%\acestep_docker_%RANDOM%.log"
cmd /c "!RD_CMD!" > "!RD_TMPLOG!" 2>&1
set "RD_EXIT=!errorlevel!"
type "!RD_TMPLOG!"
type "!RD_TMPLOG!" >> "!LOGFILE!"
del "!RD_TMPLOG!" >nul 2>&1
if !RD_EXIT! neq 0 (
    echo. >> "!LOGFILE!"
    echo ^>^>^> ОШИБКА: код выхода !RD_EXIT! >> "!LOGFILE!"
    echo.
    call :print_error "!RD_DESC! — не удалось (код: !RD_EXIT!)"
    call :print_info "Полный лог: %CYAN%!LOGFILE!%NC%"
    call :print_info "Диагностика Docker: %CYAN%docker compose logs%NC%"
    exit /b 1
)
call :print_status "!RD_DESC! — OK"
exit /b 0

:check_docker
where docker >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker не найден. Установите Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
    exit /b 1
)
for /f "tokens=*" %%v in ('docker --version 2^>nul') do set "DOCKER_VER=%%v"
call :print_status "Docker найден: !DOCKER_VER!"

docker info >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker daemon не запущен. Запустите Docker Desktop и попробуйте снова."
    exit /b 1
)
call :print_status "Docker daemon работает"
exit /b 0

:check_gpu
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :print_warn "GPU не обнаружен (nvidia-smi не найден)"
    exit /b 1
)

for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do set "GPU_NAME=%%g"
for /f "tokens=*" %%m in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader 2^>nul') do set "GPU_MEM=%%m"

set "GPU_COUNT=0"
for /f %%c in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul ^| find /c /v ""') do set "GPU_COUNT=%%c"

call :print_status "GPU: !GPU_NAME! (!GPU_MEM!) x !GPU_COUNT! шт."

docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :print_warn "nvidia-container-toolkit не настроен в Docker Desktop"
    call :print_info "Включите GPU в Docker Desktop: Settings → Resources → GPU"
    exit /b 1
)
call :print_status "nvidia-container-toolkit работает"
exit /b 0

:detect_gpu_count
set "GPU_COUNT=0"
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    for /f %%c in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul ^| find /c /v ""') do set "GPU_COUNT=%%c"
)
exit /b

:deploy_cpu
call :log_init "CPU"
echo.
echo %BOLD%═══ Режим: CPU ═══%NC%
echo.
call :run_docker "Сборка образов (CPU, без CUDA)" "docker compose -f docker-compose.cpu.yml build"
if errorlevel 1 goto :pause_on_error

call :run_docker "Запуск сервисов" "docker compose -f docker-compose.cpu.yml up -d"
if errorlevel 1 goto :pause_on_error

echo.
call :print_status "Сервисы запущены!"
echo.
echo   %BOLD%API:%NC%      http://localhost:5000
echo   %BOLD%Swagger:%NC%  http://localhost:5000/docs
echo   %BOLD%Health:%NC%   http://localhost:5000/health
echo.
echo   %YELLOW%Команды:%NC%
echo   Логи:       %CYAN%docker compose -f docker-compose.cpu.yml logs -f%NC%
echo   Стоп:       %CYAN%docker compose -f docker-compose.cpu.yml down%NC%
echo   Рестарт:    %CYAN%docker compose -f docker-compose.cpu.yml restart%NC%
echo.
call :print_warn "CPU-режим: генерация будет медленной (5-15 мин на трек)"
exit /b 0

:deploy_gpu
call :log_init "GPU"
echo.
echo %BOLD%═══ Режим: GPU ═══%NC%
echo.

call :check_gpu
if errorlevel 1 (
    call :print_error "GPU недоступен. Используйте CPU-режим или настройте GPU в Docker Desktop."
    goto :pause_on_error
)

call :run_docker "Сборка образов (CUDA 12.1)" "docker compose -f docker-compose.gpu.yml build"
if errorlevel 1 goto :pause_on_error

call :run_docker "Запуск сервисов" "docker compose -f docker-compose.gpu.yml up -d"
if errorlevel 1 goto :pause_on_error

echo.
call :print_status "Сервисы запущены!"
echo.
echo   %BOLD%API:%NC%      http://localhost:5000
echo   %BOLD%Swagger:%NC%  http://localhost:5000/docs
echo   %BOLD%Health:%NC%   http://localhost:5000/health
echo.
echo   %YELLOW%Команды:%NC%
echo   Логи:       %CYAN%docker compose -f docker-compose.gpu.yml logs -f%NC%
echo   Логи GPU:   %CYAN%docker compose -f docker-compose.gpu.yml logs -f worker%NC%
echo   Стоп:       %CYAN%docker compose -f docker-compose.gpu.yml down%NC%
echo   Рестарт:    %CYAN%docker compose -f docker-compose.gpu.yml restart%NC%
exit /b 0

:deploy_farm
echo.
echo %BOLD%═══ Режим: GPU Farm ═══%NC%
echo.

call :check_gpu
if errorlevel 1 (
    call :print_error "GPU недоступен. Установите драйверы NVIDIA и настройте GPU в Docker Desktop."
    goto :pause_on_error
)

call :detect_gpu_count
call :print_info "Обнаружено GPU: !GPU_COUNT!"

echo.
set "WORKERS=!GPU_COUNT!"
set /p "INPUT_WORKERS=%BLUE%[→]%NC% Количество воркеров [!GPU_COUNT!]: "
if not "!INPUT_WORKERS!"=="" set "WORKERS=!INPUT_WORKERS!"

set "API_WORKERS=2"
set /p "INPUT_API=%BLUE%[→]%NC% Количество API процессов [2]: "
if not "!INPUT_API!"=="" set "API_WORKERS=!INPUT_API!"

set "API_PORT=5000"
set /p "INPUT_PORT=%BLUE%[→]%NC% Порт API [5000]: "
if not "!INPUT_PORT!"=="" set "API_PORT=!INPUT_PORT!"

set "FLOWER_PORT=5555"
set /p "INPUT_FLOWER=%BLUE%[→]%NC% Порт Flower (мониторинг) [5555]: "
if not "!INPUT_FLOWER!"=="" set "FLOWER_PORT=!INPUT_FLOWER!"

set "WORKER_MEMORY=8G"
set /p "INPUT_MEM=%BLUE%[→]%NC% Лимит RAM на воркер [8G]: "
if not "!INPUT_MEM!"=="" set "WORKER_MEMORY=!INPUT_MEM!"

echo.
echo %BOLD%Конфигурация:%NC%
echo   GPU воркеров:    %CYAN%!WORKERS!%NC%
echo   API процессов:   %CYAN%!API_WORKERS!%NC%
echo   Порт API:        %CYAN%!API_PORT!%NC%
echo   Порт Flower:     %CYAN%!FLOWER_PORT!%NC%
echo   RAM / воркер:    %CYAN%!WORKER_MEMORY!%NC%
echo.

set /p "CONFIRM=%BLUE%[→]%NC% Запустить? [Y/n]: "
if /i "!CONFIRM!"=="n" (
    call :print_info "Отменено."
    exit /b 0
)

set "GPU_WORKERS=!WORKERS!"

call :log_init "Farm (workers=!WORKERS!)"

call :run_docker "Сборка образов (CUDA 12.1)" "docker compose -f docker-compose.farm.yml build"
if errorlevel 1 goto :pause_on_error

call :run_docker "Запуск фермы: !WORKERS! GPU воркер(ов)" "docker compose -f docker-compose.farm.yml up -d --scale worker=!WORKERS!"
if errorlevel 1 goto :pause_on_error

echo.
call :print_status "Ферма запущена!"
echo.
echo   %BOLD%API:%NC%        http://localhost:!API_PORT!
echo   %BOLD%Swagger:%NC%    http://localhost:!API_PORT!/docs
echo   %BOLD%Flower:%NC%     http://localhost:!FLOWER_PORT!
echo   %BOLD%Health:%NC%     http://localhost:!API_PORT!/health
echo.
echo   %YELLOW%Команды:%NC%
echo   Логи:          %CYAN%docker compose -f docker-compose.farm.yml logs -f%NC%
echo   Логи воркеров: %CYAN%docker compose -f docker-compose.farm.yml logs -f worker%NC%
echo   Масштаб:       %CYAN%set GPU_WORKERS=4 ^& docker compose -f docker-compose.farm.yml up -d --scale worker=4%NC%
echo   Стоп:          %CYAN%docker compose -f docker-compose.farm.yml down%NC%
echo.
echo   %YELLOW%Файл конфигурации: %CYAN%.env%NC%
echo   Для постоянных настроек создайте %CYAN%.env%NC% файл:
echo   %CYAN%GPU_WORKERS=!WORKERS!%NC%
echo   %CYAN%API_WORKERS=!API_WORKERS!%NC%
echo   %CYAN%API_PORT=!API_PORT!%NC%
echo   %CYAN%FLOWER_PORT=!FLOWER_PORT!%NC%
echo   %CYAN%WORKER_MEMORY=!WORKER_MEMORY!%NC%
exit /b 0

:stop_all
echo.
call :print_info "Остановка всех конфигураций..."
docker compose -f docker-compose.cpu.yml down >nul 2>&1 && call :print_status "CPU остановлен"
docker compose -f docker-compose.gpu.yml down >nul 2>&1 && call :print_status "GPU остановлен"
docker compose -f docker-compose.farm.yml down >nul 2>&1 && call :print_status "Farm остановлен"
echo.
exit /b 0

:show_menu
echo %BOLD%Выберите режим запуска:%NC%
echo.
echo   %CYAN%1%NC%) %BOLD%CPU%NC% — без видеокарты
echo      Медленная генерация, подходит для тестирования
echo.
echo   %CYAN%2%NC%) %BOLD%GPU%NC% — одна видеокарта NVIDIA
echo      Быстрая генерация, стандартный режим
echo.
echo   %CYAN%3%NC%) %BOLD%GPU Farm%NC% — несколько видеокарт
echo      Масштабирование, мониторинг через Flower
echo.
echo   %CYAN%0%NC%) Выход
echo.
exit /b

:pause_on_error
echo.
call :print_error "Развёртывание прервано из-за ошибки."
call :print_info "Изучите лог выше или файл: %CYAN%!LOGFILE!%NC%"
echo.
pause
goto :end

:main
call :print_header

if "%~1"=="cpu" (
    call :check_docker
    if not errorlevel 1 call :deploy_cpu
    goto :end_pause
)
if "%~1"=="gpu" (
    call :check_docker
    if not errorlevel 1 call :deploy_gpu
    goto :end_pause
)
if "%~1"=="farm" (
    call :check_docker
    if not errorlevel 1 call :deploy_farm
    goto :end_pause
)
if "%~1"=="stop" (
    call :check_docker
    if not errorlevel 1 call :stop_all
    goto :end_pause
)

call :check_docker
if errorlevel 1 goto :end_pause

echo.
call :check_gpu >nul 2>&1
if not errorlevel 1 (
    call :detect_gpu_count
    if !GPU_COUNT! gtr 1 (
        call :print_info "Рекомендация: GPU Farm (!GPU_COUNT! видеокарт)"
    ) else (
        call :print_info "Рекомендация: GPU (1 видеокарта)"
    )
) else (
    call :print_info "Рекомендация: CPU (GPU не обнаружен)"
)

echo.
call :show_menu

set /p "CHOICE=%BOLD%Ваш выбор [1-3]: %NC%"
if "!CHOICE!"=="1" call :deploy_cpu
if "!CHOICE!"=="2" call :deploy_gpu
if "!CHOICE!"=="3" call :deploy_farm
if "!CHOICE!"=="0" (
    echo.
    call :print_info "Выход."
)
if not "!CHOICE!"=="1" if not "!CHOICE!"=="2" if not "!CHOICE!"=="3" if not "!CHOICE!"=="0" (
    call :print_error "Неверный выбор: !CHOICE!"
)

:end_pause
echo.
pause
:end
endlocal
