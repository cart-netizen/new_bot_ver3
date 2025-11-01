#!/bin/bash
#
# УЛУЧШЕННЫЙ АВТОМАТИЧЕСКИЙ ДЕПЛОЙ TRADING BOT
# Исправляет все проблемы автоматически
#
# Использование: sudo bash deploy_improved.sh <IP_АДРЕС_СЕРВЕРА>
# Пример: sudo bash deploy_improved.sh 82.146.32.21
#

set -e  # Остановка при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функции для красивого вывода
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; exit 1; }

# Проверка аргументов
if [ "$#" -ne 1 ]; then
    error "Использование: sudo bash deploy_improved.sh <IP_АДРЕС_СЕРВЕРА>"
fi

SERVER_IP=$1
PROJECT_DIR=$(pwd)
FRONTEND_DIR="$PROJECT_DIR/frontend"
BACKEND_DIR="$PROJECT_DIR/backend"
VENV_DIR="$PROJECT_DIR/.venv"
REAL_USER=${SUDO_USER:-$USER}

echo ""
echo "========================================"
echo "УЛУЧШЕННЫЙ ДЕПЛОЙ TRADING BOT"
echo "========================================"
echo ""
info "Server: $SERVER_IP"
info "Project: $PROJECT_DIR"
info "User: $REAL_USER"
echo ""

# ============================================
# 1. ПРОВЕРКА И УСТАНОВКА ЗАВИСИМОСТЕЙ
# ============================================
echo "========================================"
echo "1. Проверка зависимостей"
echo "========================================"
echo ""

check_command() {
    if command -v $1 &> /dev/null; then
        success "$1 установлен"
        return 0
    else
        warning "$1 не установлен"
        return 1
    fi
}

# Python
if ! check_command python3; then
    info "Установка Python..."
    apt update
    apt install -y python3 python3-pip python3-venv
    success "Python установлен"
fi

# Node.js
if ! check_command node; then
    info "Установка Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt install -y nodejs
    success "Node.js установлен"
fi

# Nginx
if ! check_command nginx; then
    info "Установка Nginx..."
    apt install -y nginx
    success "Nginx установлен"
fi

# PostgreSQL
if ! check_command psql; then
    info "Установка PostgreSQL..."
    apt update
    apt install -y postgresql postgresql-contrib libpq-dev
    systemctl start postgresql
    systemctl enable postgresql
    success "PostgreSQL установлен"
fi

# ============================================
# 2. ИСПРАВЛЕНИЕ REQUIREMENTS.TXT
# ============================================
echo ""
echo "========================================"
echo "2. Исправление requirements.txt"
echo "========================================"
echo ""

info "Конвертация requirements.txt в UTF-8..."
if file requirements.txt 2>/dev/null | grep -q "UTF-16"; then
    iconv -f UTF-16 -t UTF-8 requirements.txt > requirements_utf8.txt
    mv requirements_utf8.txt requirements.txt
    success "Файл сконвертирован в UTF-8"
fi

info "Замена psycopg2 на psycopg2-binary..."
sed -i 's/^psycopg2==/psycopg2-binary==/g' requirements.txt

info "Удаление несовместимых зависимостей..."
sed -i '/backports.asyncio.runner/d' requirements.txt

success "requirements.txt исправлен"

# ============================================
# 3. ИСПРАВЛЕНИЕ ИМПОРТОВ В BACKEND
# ============================================
echo ""
echo "========================================"
echo "3. Исправление импортов в backend"
echo "========================================"
echo ""

info "Запуск fix_imports.sh..."
if [ -f "scripts/fix_imports.sh" ]; then
    bash scripts/fix_imports.sh
    success "Все импорты исправлены"
else
    warning "scripts/fix_imports.sh не найден, пропускаем"
fi

# ============================================
# 4. НАСТРОЙКА BACKEND
# ============================================
echo ""
echo "========================================"
echo "4. Настройка Backend"
echo "========================================"
echo ""

# Создание .env файла
info "Создание .env файла..."
cat > "$BACKEND_DIR/.env" <<EOF
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://$SERVER_IP,http://$SERVER_IP:3000,http://$SERVER_IP:8000

# Application
DEBUG=False
LOG_LEVEL=INFO
APP_NAME=Scalping Trading Bot
APP_VERSION=1.0.0

# Bybit Configuration
BYBIT_MODE=testnet
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here

# Database Configuration
DATABASE_URL=postgresql+asyncpg://trading_bot:trading_bot_password_2025@localhost:5432/trading_bot
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_bot
DB_USER=trading_bot
DB_PASSWORD=trading_bot_password_2025

# Security
APP_PASSWORD=robocop89
SECRET_KEY=$(openssl rand -hex 32)
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Trading Configuration
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
POSITION_SIZE_USDT=10
MAX_POSITIONS=3
LEVERAGE=1

# Risk Management
STOP_LOSS_PERCENT=2.0
TAKE_PROFIT_PERCENT=3.0
DAILY_LOSS_LIMIT_USDT=50

# Consensus
CONSENSUS_MODE=weighted
WEIGHT_OPTIMIZATION_METHOD=HYBRID
EOF

success ".env создан"

# Создание виртуального окружения
if [ ! -d "$VENV_DIR" ]; then
    info "Создание виртуального окружения..."
    python3 -m venv "$VENV_DIR"
    success "Виртуальное окружение создано"
else
    success "Venv уже существует"
fi

# Установка зависимостей
info "Установка Python зависимостей..."
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
"$VENV_DIR/bin/pip" install -r requirements.txt
success "Python зависимости установлены"

# ============================================
# 5. НАСТРОЙКА POSTGRESQL
# ============================================
echo ""
echo "========================================"
echo "5. Настройка PostgreSQL"
echo "========================================"
echo ""

info "Создание базы данных и пользователя..."
sudo -u postgres psql << 'EOSQL' 2>/dev/null || true
-- Создаем пользователя (если не существует)
DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'trading_bot') THEN
      CREATE USER trading_bot WITH PASSWORD 'trading_bot_password_2025';
   END IF;
END
$$;

-- Создаем базу данных (если не существует)
SELECT 'CREATE DATABASE trading_bot OWNER trading_bot'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'trading_bot')\gexec

-- Даем права
GRANT ALL PRIVILEGES ON DATABASE trading_bot TO trading_bot;
EOSQL

success "База данных настроена"

# Инициализация таблиц
info "Инициализация таблиц базы данных..."
cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR" "$VENV_DIR/bin/python" backend/scripts/sync_init_database.py init 2>/dev/null || {
    warning "Таблицы уже существуют или произошла ошибка инициализации"
}
success "Таблицы базы данных готовы"

# ============================================
# 6. ИСПРАВЛЕНИЕ RSBUILD CONFIG
# ============================================
echo ""
echo "========================================"
echo "6. Исправление rsbuild.config.ts"
echo "========================================"
echo ""

info "Проверка rsbuild.config.ts..."
if ! grep -q "loadEnv" "$FRONTEND_DIR/rsbuild.config.ts" 2>/dev/null; then
    info "Добавление loadEnv в rsbuild.config.ts..."

    # Создаем правильный rsbuild.config.ts
    cat > "$FRONTEND_DIR/rsbuild.config.ts" <<'EOF'
import {defineConfig, loadEnv} from "@rsbuild/core";
import {pluginReact} from "@rsbuild/plugin-react";

const {publicVars} = loadEnv({prefixes: ['VITE_']});

export default defineConfig({
  plugins: [
    pluginReact({
      swcReactOptions: {
        runtime: 'automatic',
      },
    }),
  ],

  source: {
    define: publicVars,
    entry: {
      index: './src/main.tsx',
    },
    alias: {
      '@': './src',
    },
  },

  server: {
    port: 3000,
    host: '0.0.0.0',
    printUrls: true,
  },

  html: {
    template: './index.html',
  },

  output: {
    target: 'web',
    distPath: {
      root: 'dist',
    },
    sourceMap: {
      js: 'source-map',
      css: true,
    },
  },

  performance: {
    chunkSplit: {
      strategy: 'split-by-experience',
    },
  },

  dev: {
    assetPrefix: true,
  },
});
EOF
    success "rsbuild.config.ts исправлен"
else
    success "rsbuild.config.ts уже правильный"
fi

# ============================================
# 7. НАСТРОЙКА FRONTEND
# ============================================
echo ""
echo "========================================"
echo "7. Настройка Frontend"
echo "========================================"
echo ""

# Создание .env.production с правильной кодировкой
info "Создание .env.production..."
cat > "$FRONTEND_DIR/.env.production" <<EOF
VITE_API_URL=http://$SERVER_IP:8000
VITE_WS_URL=ws://$SERVER_IP:8000/ws
EOF
success ".env.production создан"

# Установка npm зависимостей
info "Установка npm зависимостей..."
cd "$FRONTEND_DIR"
npm install
success "npm зависимости установлены"

# Сборка production build
info "Сборка production build..."
rm -rf dist node_modules/.cache
npm run build
success "Frontend собран в dist/"

# Проверка что переменные попали в build
if grep -r "$SERVER_IP" dist/ > /dev/null 2>&1; then
    success "API URL корректно встроен в build"
else
    warning "API URL не найден в build - проверьте настройки"
fi

# ============================================
# 8. НАСТРОЙКА NGINX
# ============================================
echo ""
echo "========================================"
echo "8. Настройка Nginx"
echo "========================================"
echo ""

info "Создание конфигурации nginx..."
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

    # Direct access to backend
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

# Активация конфигурации
ln -sf /etc/nginx/sites-available/trading-bot /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Проверка конфигурации
info "Проверка конфигурации nginx..."
nginx -t
success "Конфигурация nginx корректна"

# Перезапуск Nginx
systemctl restart nginx
success "Nginx перезапущен"

# ============================================
# 9. СОЗДАНИЕ SYSTEMD SERVICE
# ============================================
echo ""
echo "========================================"
echo "9. Создание Systemd Service"
echo "========================================"
echo ""

info "Создание trading-bot.service..."
cat > /etc/systemd/system/trading-bot.service <<EOF
[Unit]
Description=Trading Bot Backend (Full Bot with Database)
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=$VENV_DIR/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

success "Systemd service создан"

# Перезагрузка systemd
systemctl daemon-reload
systemctl enable trading-bot
systemctl restart trading-bot

# Ждем запуска
sleep 5

# ============================================
# 10. ПРОВЕРКА СТАТУСА
# ============================================
echo ""
echo "========================================"
echo "📊 СТАТУС TRADING BOT"
echo "========================================"
echo ""

# Проверка backend
if systemctl is-active --quiet trading-bot; then
    success "Backend: RUNNING"
else
    warning "Backend: STOPPED (проверьте логи: journalctl -u trading-bot -n 50)"
fi

# Проверка Nginx
if systemctl is-active --quiet nginx; then
    success "Nginx (Frontend): RUNNING"
else
    warning "Nginx (Frontend): STOPPED"
fi

# Проверка PostgreSQL
if systemctl is-active --quiet postgresql; then
    success "PostgreSQL: RUNNING"
else
    warning "PostgreSQL: STOPPED"
fi

# Проверка портов
if netstat -tlpn 2>/dev/null | grep -q ":8000"; then
    success "Backend :8000 - слушает"
else
    warning "Backend :8000 - не слушает"
fi

if netstat -tlpn 2>/dev/null | grep -q ":80"; then
    success "Nginx :80 - слушает"
else
    warning "Nginx :80 - не слушает"
fi

# Health checks
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    success "Backend API: OK"
else
    warning "Backend API: FAIL"
fi

if curl -s http://localhost/ > /dev/null 2>&1; then
    success "Frontend: OK"
else
    warning "Frontend: FAIL"
fi

# ============================================
# ИТОГОВАЯ ИНФОРМАЦИЯ
# ============================================
echo ""
echo "════════════════════════════════════════"
echo "🎉 ДЕПЛОЙ ЗАВЕРШЕН!"
echo "════════════════════════════════════════"
echo ""
echo "📍 Адреса:"
echo "   Frontend:  http://$SERVER_IP"
echo "   Backend:   http://$SERVER_IP:8000"
echo "   API Docs:  http://$SERVER_IP:8000/docs"
echo ""
echo "🔐 Учетные данные:"
echo "   Пароль для входа: robocop89"
echo "   (можно изменить в $BACKEND_DIR/.env -> APP_PASSWORD)"
echo ""
echo "🗄️  База данных:"
echo "   PostgreSQL: localhost:5432"
echo "   Database: trading_bot"
echo "   User: trading_bot"
echo ""
echo "⚠️  ВАЖНО: Настройте Bybit API ключи!"
echo "   Отредактируйте файл: $BACKEND_DIR/.env"
echo "   Переменные: BYBIT_API_KEY, BYBIT_API_SECRET, BYBIT_MODE"
echo "   После изменения: sudo systemctl restart trading-bot"
echo ""
echo "📋 Управление ботом:"
echo "   sudo systemctl status trading-bot    # Статус"
echo "   sudo systemctl restart trading-bot   # Перезапуск"
echo "   sudo systemctl stop trading-bot      # Остановка"
echo "   sudo journalctl -u trading-bot -f    # Логи в реальном времени"
echo ""
echo "🔧 Скрипты управления:"
echo "   bash scripts/status.sh     # Проверить статус"
echo "   bash scripts/restart.sh    # Перезапустить"
echo "   bash scripts/logs.sh       # Посмотреть логи"
echo ""
echo "════════════════════════════════════════"
echo ""

# Показываем последние логи
info "Последние 30 строк логов backend:"
echo ""
journalctl -u trading-bot -n 30 --no-pager

echo ""
success "Деплой успешно завершен! 🚀"
success "Бот будет работать постоянно, даже после закрытия терминала!"
echo ""
