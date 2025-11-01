#!/bin/bash
#
# Автоматический скрипт деплоя Trading Bot
# Запускает backend и frontend одновременно
#
# Использование:
#   sudo bash deploy.sh 82.146.32.21
#

set -e  # Остановка при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для вывода
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Проверка что запущен как root
if [[ $EUID -ne 0 ]]; then
   print_error "Этот скрипт должен быть запущен с правами root (sudo)"
   exit 1
fi

# Получаем реального пользователя (не root)
REAL_USER=${SUDO_USER:-$USER}
REAL_HOME=$(eval echo ~$REAL_USER)

# IP адрес или домен
SERVER_IP=${1:-""}
if [[ -z "$SERVER_IP" ]]; then
    print_error "Укажите IP адрес или домен сервера"
    echo "Использование: sudo bash deploy.sh <IP_OR_DOMAIN>"
    echo "Пример: sudo bash deploy.sh 82.146.32.21"
    exit 1
fi

# Определяем директории
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
VENV_DIR="$PROJECT_DIR/.venv"

print_header "АВТОМАТИЧЕСКИЙ ДЕПЛОЙ TRADING BOT"
print_info "Server: $SERVER_IP"
print_info "Project: $PROJECT_DIR"
print_info "User: $REAL_USER"

# ============================================
# 1. Проверка зависимостей
# ============================================
print_header "1. Проверка зависимостей"

check_command() {
    if command -v $1 &> /dev/null; then
        print_success "$1 установлен"
        return 0
    else
        print_warning "$1 не найден, устанавливаю..."
        return 1
    fi
}

# Python 3
if ! check_command python3; then
    apt update
    apt install -y python3 python3-pip python3-venv
fi

# Node.js
if ! check_command node; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt install -y nodejs
fi

# Nginx
if ! check_command nginx; then
    apt install -y nginx
fi

# ============================================
# 2. Настройка Backend
# ============================================
print_header "2. Настройка Backend"

# Создаем/обновляем .env
print_info "Создание .env файла..."
cat > "$BACKEND_DIR/.env" <<EOF
# API Server
API_HOST=0.0.0.0
API_PORT=8000

# CORS - разрешаем доступ с фронтенда
CORS_ORIGINS=http://$SERVER_IP,http://$SERVER_IP:3000,http://$SERVER_IP:8000,http://localhost:3000,http://localhost:8000

# Debug (отключаем для production)
DEBUG=False

# Trading mode
BYBIT_MODE=testnet

# Database (если используется)
DATABASE_URL=sqlite:///./trading_bot.db

# Logging
LOG_LEVEL=INFO
EOF
chown $REAL_USER:$REAL_USER "$BACKEND_DIR/.env"
print_success ".env создан"

# Создаем/активируем venv
if [ ! -d "$VENV_DIR" ]; then
    print_info "Создание виртуального окружения..."
    sudo -u $REAL_USER python3 -m venv "$VENV_DIR"
    print_success "Venv создан"
else
    print_success "Venv уже существует"
fi

# Устанавливаем зависимости
print_info "Установка Python зависимостей..."
sudo -u $REAL_USER "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u $REAL_USER "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
print_success "Python зависимости установлены"

# ============================================
# 3. Настройка Frontend
# ============================================
print_header "3. Настройка Frontend"

# Создаем .env.production
print_info "Создание .env.production..."
cat > "$FRONTEND_DIR/.env.production" <<EOF
# API URL - используем IP сервера
VITE_API_URL=http://$SERVER_IP:8000
VITE_WS_URL=ws://$SERVER_IP:8000/ws

# Или через nginx (если настроен)
# VITE_API_URL=http://$SERVER_IP/api
# VITE_WS_URL=ws://$SERVER_IP/ws
EOF
chown $REAL_USER:$REAL_USER "$FRONTEND_DIR/.env.production"
print_success ".env.production создан"

# Устанавливаем npm зависимости
print_info "Установка npm зависимостей..."
cd "$FRONTEND_DIR"
sudo -u $REAL_USER npm install
print_success "npm зависимости установлены"

# Собираем production build
print_info "Сборка production build..."
sudo -u $REAL_USER npm run build
print_success "Frontend собран в dist/"

# ============================================
# 4. Настройка Nginx
# ============================================
print_header "4. Настройка Nginx"

print_info "Создание конфигурации nginx..."
cat > /etc/nginx/sites-available/trading-bot <<EOF
server {
    listen 80;
    server_name $SERVER_IP;

    # Логи
    access_log /var/log/nginx/trading-bot-access.log;
    error_log /var/log/nginx/trading-bot-error.log;

    # Frontend (статика)
    location / {
        root $FRONTEND_DIR/dist;
        try_files \$uri \$uri/ /index.html;

        # CORS headers для статики
        add_header Access-Control-Allow-Origin "*";

        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }

    # Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;

        # Таймауты
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Direct access to backend (альтернативный путь)
    location ~ ^/(health|docs|redoc|openapi.json) {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    # WebSocket
    location /ws {
        proxy_pass http://127.0.0.1:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 86400;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
    }
}
EOF

# Создаем симлинк
if [ -f /etc/nginx/sites-enabled/trading-bot ]; then
    rm /etc/nginx/sites-enabled/trading-bot
fi
ln -s /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/

# Удаляем default конфиг если есть
if [ -f /etc/nginx/sites-enabled/default ]; then
    print_info "Удаление default конфига nginx"
    rm /etc/nginx/sites-enabled/default
fi

# Проверяем конфигурацию
print_info "Проверка конфигурации nginx..."
nginx -t
print_success "Конфигурация nginx корректна"

# ============================================
# 5. Создание Systemd Service для Backend
# ============================================
print_header "5. Создание Systemd Service"

print_info "Создание trading-bot.service..."
cat > /etc/systemd/system/trading-bot.service <<EOF
[Unit]
Description=Trading Bot Backend (FastAPI)
After=network.target

[Service]
Type=simple
User=$REAL_USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$VENV_DIR/bin"
ExecStart=$VENV_DIR/bin/python -m uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

# Логи
StandardOutput=append:/var/log/trading-bot.log
StandardError=append:/var/log/trading-bot-error.log

# Безопасность
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Создаем файлы логов
touch /var/log/trading-bot.log
touch /var/log/trading-bot-error.log
chown $REAL_USER:$REAL_USER /var/log/trading-bot.log
chown $REAL_USER:$REAL_USER /var/log/trading-bot-error.log

print_success "Systemd service создан"

# ============================================
# 6. Настройка Firewall
# ============================================
print_header "6. Настройка Firewall"

if command -v ufw &> /dev/null; then
    print_info "Настройка UFW..."

    # Разрешаем порты
    ufw allow 22/tcp comment 'SSH'
    ufw allow 80/tcp comment 'HTTP'
    ufw allow 443/tcp comment 'HTTPS'
    ufw allow 8000/tcp comment 'Backend API'

    # Включаем UFW если отключен
    ufw --force enable

    print_success "Firewall настроен"
    ufw status
else
    print_warning "UFW не установлен, пропускаю настройку firewall"
fi

# ============================================
# 7. Запуск сервисов
# ============================================
print_header "7. Запуск сервисов"

# Перезагружаем systemd
print_info "Перезагрузка systemd daemon..."
systemctl daemon-reload

# Останавливаем старые процессы
print_info "Остановка старых процессов..."
systemctl stop trading-bot 2>/dev/null || true
systemctl stop nginx 2>/dev/null || true

# Запускаем Backend
print_info "Запуск Backend..."
systemctl start trading-bot
systemctl enable trading-bot
sleep 2

# Проверяем статус backend
if systemctl is-active --quiet trading-bot; then
    print_success "Backend запущен"
else
    print_error "Backend не запустился!"
    journalctl -u trading-bot -n 20 --no-pager
    exit 1
fi

# Запускаем Nginx (Frontend)
print_info "Запуск Nginx (Frontend)..."
systemctl start nginx
systemctl enable nginx

# Проверяем статус nginx
if systemctl is-active --quiet nginx; then
    print_success "Nginx запущен"
else
    print_error "Nginx не запустился!"
    journalctl -u nginx -n 20 --no-pager
    exit 1
fi

# ============================================
# 8. Проверка работоспособности
# ============================================
print_header "8. Проверка работоспособности"

# Ждем пока backend прогрузится
print_info "Ожидание инициализации backend..."
sleep 3

# Проверяем Backend
print_info "Проверка Backend API..."
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Backend API отвечает"
else
    print_warning "Backend API не отвечает на /health"
    print_info "Проверьте логи: sudo journalctl -u trading-bot -f"
fi

# Проверяем Frontend
print_info "Проверка Frontend..."
if curl -f -s http://localhost/ > /dev/null 2>&1; then
    print_success "Frontend доступен"
else
    print_warning "Frontend не отвечает"
    print_info "Проверьте логи: sudo tail -f /var/log/nginx/trading-bot-error.log"
fi

# ============================================
# 9. Итоговая информация
# ============================================
print_header "✅ ДЕПЛОЙ ЗАВЕРШЕН!"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        🚀 Trading Bot успешно развернут!              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📍 Доступ к приложению:${NC}"
echo -e "   Frontend:  ${GREEN}http://$SERVER_IP${NC}"
echo -e "   Backend:   ${GREEN}http://$SERVER_IP:8000${NC}"
echo -e "   API Docs:  ${GREEN}http://$SERVER_IP:8000/docs${NC}"
echo -e "   Health:    ${GREEN}http://$SERVER_IP:8000/health${NC}"
echo ""
echo -e "${BLUE}📊 Управление сервисами:${NC}"
echo -e "   Статус:           ${YELLOW}sudo systemctl status trading-bot${NC}"
echo -e "   Перезапуск:       ${YELLOW}sudo systemctl restart trading-bot${NC}"
echo -e "   Остановка:        ${YELLOW}sudo systemctl stop trading-bot${NC}"
echo -e "   Логи (realtime):  ${YELLOW}sudo journalctl -u trading-bot -f${NC}"
echo -e "   Логи (последние): ${YELLOW}sudo tail -f /var/log/trading-bot.log${NC}"
echo ""
echo -e "${BLUE}🌐 Nginx:${NC}"
echo -e "   Статус:           ${YELLOW}sudo systemctl status nginx${NC}"
echo -e "   Перезапуск:       ${YELLOW}sudo systemctl restart nginx${NC}"
echo -e "   Логи:             ${YELLOW}sudo tail -f /var/log/nginx/trading-bot-access.log${NC}"
echo ""
echo -e "${BLUE}🔄 Быстрый перезапуск всего:${NC}"
echo -e "   ${YELLOW}sudo systemctl restart trading-bot nginx${NC}"
echo ""
echo -e "${BLUE}📝 Полезные команды:${NC}"
echo -e "   Проверка портов:  ${YELLOW}sudo netstat -tulpn | grep -E '(8000|80)'${NC}"
echo -e "   Firewall:         ${YELLOW}sudo ufw status verbose${NC}"
echo -e "   Процессы:         ${YELLOW}ps aux | grep -E '(uvicorn|nginx)'${NC}"
echo ""
echo -e "${GREEN}✅ Всё работает! Откройте браузер и перейдите на:${NC}"
echo -e "${GREEN}   http://$SERVER_IP${NC}"
echo ""

# Показываем статус сервисов
systemctl status trading-bot --no-pager -l | head -15
systemctl status nginx --no-pager -l | head -10
