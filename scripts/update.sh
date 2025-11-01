#!/bin/bash
#
# Обновление кода и перезапуск
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

if [[ $EUID -ne 0 ]]; then
   print_error "Запустите с sudo: sudo bash scripts/update.sh"
   exit 1
fi

REAL_USER=${SUDO_USER:-$USER}
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_DIR/frontend"
BACKEND_DIR="$PROJECT_DIR/backend"

echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${GREEN}🔄 ОБНОВЛЕНИЕ TRADING BOT${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Git pull
print_info "Получение последних изменений из git..."
cd "$PROJECT_DIR"
if [ -d .git ]; then
    sudo -u $REAL_USER git pull
    print_success "Код обновлен"
else
    print_warning "Не git репозиторий, пропускаю git pull"
fi

# Обновление Python зависимостей
print_info "Обновление Python зависимостей..."
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    sudo -u $REAL_USER "$PROJECT_DIR/.venv/bin/pip" install --upgrade -r "$PROJECT_DIR/requirements.txt"
    print_success "Python зависимости обновлены"
fi

# Обновление npm зависимостей
print_info "Обновление npm зависимостей..."
cd "$FRONTEND_DIR"
sudo -u $REAL_USER npm install
print_success "npm зависимости обновлены"

# Пересборка frontend
print_info "Пересборка frontend..."
sudo -u $REAL_USER npm run build
print_success "Frontend пересобран"

# Перезапуск сервисов
print_info "Перезапуск сервисов..."
systemctl restart trading-bot
systemctl reload nginx

sleep 2

if systemctl is-active --quiet trading-bot; then
    print_success "Backend перезапущен"
else
    print_error "Backend не запустился!"
    journalctl -u trading-bot -n 20 --no-pager
    exit 1
fi

if systemctl is-active --quiet nginx; then
    print_success "Nginx перезагружен"
fi

print_success "Обновление завершено!"

# Показываем статус
bash "$SCRIPT_DIR/status.sh"
