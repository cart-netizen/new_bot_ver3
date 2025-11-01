#!/bin/bash
#
# Проверка статуса всех компонентов
#

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    if systemctl is-active --quiet $1; then
        echo -e "${GREEN}✅ $2: RUNNING${NC}"
    else
        echo -e "${RED}❌ $2: STOPPED${NC}"
    fi
}

echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${GREEN}📊 СТАТУС TRADING BOT${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Статус сервисов
echo -e "${BLUE}Сервисы:${NC}"
print_status trading-bot "Backend"
print_status nginx "Nginx (Frontend)"
echo ""

# Порты
echo -e "${BLUE}Порты:${NC}"
if netstat -tulpn 2>/dev/null | grep -q ":8000"; then
    echo -e "${GREEN}✅ Backend :8000 - слушает${NC}"
else
    echo -e "${RED}❌ Backend :8000 - не слушает${NC}"
fi

if netstat -tulpn 2>/dev/null | grep -q ":80"; then
    echo -e "${GREEN}✅ Nginx :80 - слушает${NC}"
else
    echo -e "${RED}❌ Nginx :80 - не слушает${NC}"
fi
echo ""

# API Health Check
echo -e "${BLUE}Health Checks:${NC}"
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    echo -e "${GREEN}✅ Backend API: OK${NC}"
    echo "   Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}❌ Backend API: FAIL${NC}"
fi

if curl -f -s http://localhost/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend: OK${NC}"
else
    echo -e "${RED}❌ Frontend: FAIL${NC}"
fi
echo ""

# Процессы
echo -e "${BLUE}Процессы:${NC}"
ps aux | grep -E "(uvicorn|nginx: master)" | grep -v grep | awk '{printf "  PID: %-6s CPU: %-5s MEM: %-5s CMD: %s\n", $2, $3"%", $4"%", $11" "$12" "$13}'
echo ""

# Последние ошибки
echo -e "${BLUE}Последние ошибки Backend:${NC}"
sudo journalctl -u trading-bot --no-pager -n 3 --priority=err --since "5 minutes ago" 2>/dev/null || echo "  Нет ошибок за последние 5 минут"
echo ""

# Использование дисков
echo -e "${BLUE}Использование диска:${NC}"
df -h / | tail -1 | awk '{printf "  Total: %-6s Used: %-6s Available: %-6s Use%%: %s\n", $2, $3, $4, $5}'
echo ""

echo -e "${GREEN}Для просмотра логов: ${YELLOW}bash scripts/logs.sh [backend|nginx|all]${NC}"
