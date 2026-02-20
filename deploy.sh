#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

LOGFILE="deploy.log"

print_header() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  ${BOLD}ACE-Step Music Generation API — Docker Deploy${NC}       ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warn()   { echo -e "${YELLOW}[!]${NC} $1"; }
print_error()  { echo -e "${RED}[✗]${NC} $1"; }
print_info()   { echo -e "${BLUE}[→]${NC} $1"; }

log_init() {
    echo "=== ACE-Step Deploy Log ===" > "${LOGFILE}"
    echo "Дата: $(date)" >> "${LOGFILE}"
    echo "Режим: $1" >> "${LOGFILE}"
    echo "===========================" >> "${LOGFILE}"
    echo "" >> "${LOGFILE}"
}

run_docker() {
    local description="$1"
    shift
    print_info "${description}"
    echo "" >> "${LOGFILE}"
    echo ">>> ${description}" >> "${LOGFILE}"
    echo ">>> Команда: $*" >> "${LOGFILE}"
    echo "" >> "${LOGFILE}"

    "$@" 2>&1 | tee -a "${LOGFILE}"
    local exit_code=${PIPESTATUS[0]}

    if [[ ${exit_code} -ne 0 ]]; then
        echo "" >> "${LOGFILE}"
        echo ">>> ОШИБКА: код выхода ${exit_code}" >> "${LOGFILE}"
        echo ""
        print_error "${description} — не удалось (код: ${exit_code})"
        echo ""
        echo -e "${YELLOW}═══ Последние строки лога: ═══${NC}"
        tail -30 "${LOGFILE}" | grep -v "^>>>"
        echo -e "${YELLOW}══════════════════════════════${NC}"
        echo ""
        print_info "Полный лог: ${CYAN}${LOGFILE}${NC}"
        print_info "Диагностика Docker: ${CYAN}docker compose logs${NC}"
        return 1
    fi
    return 0
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker не найден. Установите Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_status "Docker найден: $(docker --version | head -1)"

    if ! docker info &> /dev/null 2>&1; then
        print_error "Docker daemon не запущен. Запустите Docker и попробуйте снова."
        exit 1
    fi
    print_status "Docker daemon работает"
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        local gpu_mem
        gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
        local gpu_count
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        print_status "GPU: ${gpu_name} (${gpu_mem}) × ${gpu_count} шт."

        if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            print_status "nvidia-container-toolkit работает"
            return 0
        else
            print_warn "nvidia-container-toolkit не настроен"
            print_info "Установка: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            return 1
        fi
    else
        print_warn "GPU не обнаружен (nvidia-smi не найден)"
        return 1
    fi
}

detect_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l
    else
        echo "0"
    fi
}

show_menu() {
    echo -e "${BOLD}Выберите режим запуска:${NC}"
    echo ""
    echo -e "  ${CYAN}1${NC}) ${BOLD}CPU${NC} — без видеокарты"
    echo -e "     Медленная генерация, подходит для тестирования"
    echo ""
    echo -e "  ${CYAN}2${NC}) ${BOLD}GPU${NC} — одна видеокарта NVIDIA"
    echo -e "     Быстрая генерация, стандартный режим"
    echo ""
    echo -e "  ${CYAN}3${NC}) ${BOLD}GPU Farm${NC} — несколько видеокарт"
    echo -e "     Масштабирование, мониторинг через Flower"
    echo ""
    echo -e "  ${CYAN}0${NC}) Выход"
    echo ""
}

deploy_cpu() {
    log_init "CPU"
    echo ""
    echo -e "${BOLD}═══ Режим: CPU ═══${NC}"
    echo ""

    if ! run_docker "Сборка образов (CPU, без CUDA)..." docker compose -f docker-compose.cpu.yml build; then
        return 1
    fi

    if ! run_docker "Запуск сервисов..." docker compose -f docker-compose.cpu.yml up -d; then
        return 1
    fi

    echo ""
    print_status "Сервисы запущены!"
    echo ""
    echo -e "  ${BOLD}API:${NC}      http://localhost:5000"
    echo -e "  ${BOLD}Swagger:${NC}  http://localhost:5000/docs"
    echo -e "  ${BOLD}Health:${NC}   http://localhost:5000/health"
    echo ""
    echo -e "  ${YELLOW}Команды:${NC}"
    echo -e "  Логи:       ${CYAN}docker compose -f docker-compose.cpu.yml logs -f${NC}"
    echo -e "  Стоп:       ${CYAN}docker compose -f docker-compose.cpu.yml down${NC}"
    echo -e "  Рестарт:    ${CYAN}docker compose -f docker-compose.cpu.yml restart${NC}"
    echo ""
    print_warn "CPU-режим: генерация будет медленной (5-15 мин на трек)"
}

deploy_gpu() {
    log_init "GPU"
    echo ""
    echo -e "${BOLD}═══ Режим: GPU ═══${NC}"
    echo ""

    if ! check_gpu; then
        print_error "GPU недоступен. Используйте CPU-режим или настройте nvidia-container-toolkit."
        return 1
    fi

    if ! run_docker "Сборка образов (CUDA 12.1)..." docker compose -f docker-compose.gpu.yml build; then
        return 1
    fi

    if ! run_docker "Запуск сервисов..." docker compose -f docker-compose.gpu.yml up -d; then
        return 1
    fi

    echo ""
    print_status "Сервисы запущены!"
    echo ""
    echo -e "  ${BOLD}API:${NC}      http://localhost:5000"
    echo -e "  ${BOLD}Swagger:${NC}  http://localhost:5000/docs"
    echo -e "  ${BOLD}Health:${NC}   http://localhost:5000/health"
    echo ""
    echo -e "  ${YELLOW}Команды:${NC}"
    echo -e "  Логи:       ${CYAN}docker compose -f docker-compose.gpu.yml logs -f${NC}"
    echo -e "  Логи GPU:   ${CYAN}docker compose -f docker-compose.gpu.yml logs -f worker${NC}"
    echo -e "  Стоп:       ${CYAN}docker compose -f docker-compose.gpu.yml down${NC}"
    echo -e "  Рестарт:    ${CYAN}docker compose -f docker-compose.gpu.yml restart${NC}"
}

deploy_farm() {
    echo ""
    echo -e "${BOLD}═══ Режим: GPU Farm ═══${NC}"
    echo ""

    if ! check_gpu; then
        print_error "GPU недоступен. Установите драйверы NVIDIA и nvidia-container-toolkit."
        return 1
    fi

    local gpu_count
    gpu_count=$(detect_gpu_count)
    print_info "Обнаружено GPU: ${gpu_count}"

    local workers="${gpu_count}"
    echo ""
    read -p "$(echo -e "${BLUE}[→]${NC} Количество воркеров [${gpu_count}]: ")" input_workers
    workers="${input_workers:-$gpu_count}"

    local api_workers="2"
    read -p "$(echo -e "${BLUE}[→]${NC} Количество API процессов [2]: ")" input_api
    api_workers="${input_api:-2}"

    local api_port="5000"
    read -p "$(echo -e "${BLUE}[→]${NC} Порт API [5000]: ")" input_port
    api_port="${input_port:-5000}"

    local flower_port="5555"
    read -p "$(echo -e "${BLUE}[→]${NC} Порт Flower (мониторинг) [5555]: ")" input_flower
    flower_port="${input_flower:-5555}"

    local worker_mem="8G"
    read -p "$(echo -e "${BLUE}[→]${NC} Лимит RAM на воркер [8G]: ")" input_mem
    worker_mem="${input_mem:-8G}"

    echo ""
    echo -e "${BOLD}Конфигурация:${NC}"
    echo -e "  GPU воркеров:    ${CYAN}${workers}${NC}"
    echo -e "  API процессов:   ${CYAN}${api_workers}${NC}"
    echo -e "  Порт API:        ${CYAN}${api_port}${NC}"
    echo -e "  Порт Flower:     ${CYAN}${flower_port}${NC}"
    echo -e "  RAM / воркер:    ${CYAN}${worker_mem}${NC}"
    echo ""

    read -p "$(echo -e "${BLUE}[→]${NC} Запустить? [Y/n]: ")" confirm
    if [[ "${confirm}" =~ ^[Nn] ]]; then
        print_info "Отменено."
        return 0
    fi

    export GPU_WORKERS="${workers}"
    export API_WORKERS="${api_workers}"
    export API_PORT="${api_port}"
    export FLOWER_PORT="${flower_port}"
    export WORKER_MEMORY="${worker_mem}"

    log_init "Farm (workers=${workers})"

    if ! run_docker "Сборка образов (CUDA 12.1)..." docker compose -f docker-compose.farm.yml build; then
        return 1
    fi

    if ! run_docker "Запуск фермы: ${workers} GPU воркер(ов)..." docker compose -f docker-compose.farm.yml up -d --scale worker="${workers}"; then
        return 1
    fi

    echo ""
    print_status "Ферма запущена!"
    echo ""
    echo -e "  ${BOLD}API:${NC}        http://localhost:${api_port}"
    echo -e "  ${BOLD}Swagger:${NC}    http://localhost:${api_port}/docs"
    echo -e "  ${BOLD}Flower:${NC}     http://localhost:${flower_port}"
    echo -e "  ${BOLD}Health:${NC}     http://localhost:${api_port}/health"
    echo ""
    echo -e "  ${YELLOW}Команды:${NC}"
    echo -e "  Логи:          ${CYAN}docker compose -f docker-compose.farm.yml logs -f${NC}"
    echo -e "  Логи воркеров: ${CYAN}docker compose -f docker-compose.farm.yml logs -f worker${NC}"
    echo -e "  Масштаб:       ${CYAN}GPU_WORKERS=4 docker compose -f docker-compose.farm.yml up -d --scale worker=4${NC}"
    echo -e "  Стоп:          ${CYAN}docker compose -f docker-compose.farm.yml down${NC}"
    echo ""
    echo -e "  ${YELLOW}Файл конфигурации: ${CYAN}.env${NC}"
    echo -e "  Для постоянных настроек создайте ${CYAN}.env${NC} файл:"
    echo -e "  ${CYAN}GPU_WORKERS=${workers}${NC}"
    echo -e "  ${CYAN}API_WORKERS=${api_workers}${NC}"
    echo -e "  ${CYAN}API_PORT=${api_port}${NC}"
    echo -e "  ${CYAN}FLOWER_PORT=${flower_port}${NC}"
    echo -e "  ${CYAN}WORKER_MEMORY=${worker_mem}${NC}"
}

stop_all() {
    echo ""
    print_info "Остановка всех конфигураций..."
    docker compose -f docker-compose.cpu.yml down 2>/dev/null && print_status "CPU остановлен" || true
    docker compose -f docker-compose.gpu.yml down 2>/dev/null && print_status "GPU остановлен" || true
    docker compose -f docker-compose.farm.yml down 2>/dev/null && print_status "Farm остановлен" || true
    echo ""
}

handle_args() {
    case "${1}" in
        cpu)    check_docker; deploy_cpu ;;
        gpu)    check_docker; deploy_gpu ;;
        farm)   check_docker; deploy_farm ;;
        stop)   check_docker; stop_all ;;
        *)      return 1 ;;
    esac
    return 0
}

main() {
    print_header

    if [[ $# -gt 0 ]]; then
        if handle_args "$@"; then
            exit 0
        fi
    fi

    check_docker
    echo ""

    if check_gpu 2>/dev/null; then
        local gpu_count
        gpu_count=$(detect_gpu_count)
        if [[ "${gpu_count}" -gt 1 ]]; then
            print_info "Рекомендация: GPU Farm (${gpu_count} видеокарт)"
        else
            print_info "Рекомендация: GPU (1 видеокарта)"
        fi
    else
        print_info "Рекомендация: CPU (GPU не обнаружен)"
    fi

    echo ""
    show_menu

    read -p "$(echo -e "${BOLD}Ваш выбор [1-3]: ${NC}")" choice
    case "${choice}" in
        1) deploy_cpu ;;
        2) deploy_gpu ;;
        3) deploy_farm ;;
        0) echo ""; print_info "Выход."; exit 0 ;;
        *) print_error "Неверный выбор: ${choice}"; exit 1 ;;
    esac

    echo ""
}

main "$@"
