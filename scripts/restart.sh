#!/bin/bash
#
# Быстрый перезапуск Backend и Frontend
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

if [[ $EUID -ne 0 ]]; then
   print_error "Запустите с sudo: sudo bash scripts/restart.sh"
   exit 1
fi

print_info "Перезапуск Trading Bot..."

# Перезапуск backend
print_info "Останавливаю backend..."
systemctl stop trading-bot

print_info "Запускаю backend..."
systemctl start trading-bot
sleep 2

if systemctl is-active --quiet trading-bot; then
    print_success "Backend перезапущен"
else
    print_error "Backend не запустился!"
    journalctl -u trading-bot -n 20 --no-pager
    exit 1
fi

# Перезапуск nginx
print_info "Перезагружаю nginx..."
systemctl reload nginx

if systemctl is-active --quiet nginx; then
    print_success "Nginx перезагружен"
else
    print_error "Nginx не работает!"
    exit 1
fi

print_success "Всё перезапущено!"

# Показываем статус
echo ""
systemctl status trading-bot --no-pager -l | head -10
systemctl status nginx --no-pager -l | head -5
