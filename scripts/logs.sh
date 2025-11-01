#!/bin/bash
#
# Просмотр логов в реальном времени
#

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${GREEN}📝 ЛОГИ TRADING BOT${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

SERVICE=${1:-"trading-bot"}

case $SERVICE in
    backend|bot|trading-bot)
        echo -e "${GREEN}Логи Backend (Ctrl+C для выхода):${NC}"
        sudo journalctl -u trading-bot -f --no-pager
        ;;
    nginx|frontend)
        echo -e "${GREEN}Логи Nginx (Ctrl+C для выхода):${NC}"
        sudo tail -f /var/log/nginx/trading-bot-access.log /var/log/nginx/trading-bot-error.log
        ;;
    all|both)
        echo -e "${GREEN}Все логи (Ctrl+C для выхода):${NC}"
        sudo journalctl -u trading-bot -u nginx -f --no-pager
        ;;
    *)
        echo "Использование: bash scripts/logs.sh [backend|nginx|all]"
        echo ""
        echo "Примеры:"
        echo "  bash scripts/logs.sh backend   # Логи backend"
        echo "  bash scripts/logs.sh nginx     # Логи nginx"
        echo "  bash scripts/logs.sh all       # Все логи"
        exit 1
        ;;
esac
