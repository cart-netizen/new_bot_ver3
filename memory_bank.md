Реализовано. 

# Торговые пары
TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT
```

**Важно:**
- Сгенерируйте секретный ключ: `openssl rand -hex 32`
- Используйте сильный пароль для APP_PASSWORD
- Для начала используйте testnet режим

## 🚀 Запуск

### Запуск бэкенда

```bash
python main.py
```

Сервер запустится на `http://localhost:8000`

### Проверка работы

1. Откройте документацию API: `http://localhost:8000/docs`
2. Проверьте статус: `http://localhost:8000/`

## 📡 API Эндпоинты

### Аутентификация

- `POST /auth/login` - Вход в систему
- `POST /auth/change-password` - Изменение пароля
- `GET /auth/verify` - Проверка токена

### Управление ботом

- `GET /bot/status` - Получение статуса бота
- `POST /bot/start` - Запуск бота
- `POST /bot/stop` - Остановка бота
- `GET /bot/config` - Получение конфигурации

### Рыночные данные

- `GET /data/pairs` - Список торговых пар
- `GET /data/orderbook/{symbol}` - Стакан для пары
- `GET /data/metrics/{symbol}` - Метрики для пары
- `GET /data/metrics` - Метрики для всех пар

### Торговля

- `GET /trading/signals` - Торговые сигналы
- `GET /trading/balance` - Баланс счета
- `GET /trading/positions` - Открытые позиции
- `GET /trading/risk-status` - Статус риска
- `GET /trading/execution-stats` - Статистика исполнения

### WebSocket

- `WS /ws` - WebSocket соединение для реалтайм обновлений

## 🔧 Настройка стратегии

В файле `.env` можно настроить параметры стратегии:

```env
# Порог дисбаланса для покупки (0.0-1.0)
IMBALANCE_BUY_THRESHOLD=0.75

# Порог дисбаланса для продажи (0.0-1.0)
IMBALANCE_SELL_THRESHOLD=0.25

# Минимальный объем для кластера
MIN_CLUSTER_VOLUME=10000

# Максимальное количество позиций
MAX_OPEN_POSITIONS=5

# Максимальная экспозиция в USDT
MAX_EXPOSURE_USDT=10000

# Минимальный размер ордера в USDT
MIN_ORDER_SIZE_USDT=5
```

## 📊 Структура проекта

```
backend/
├── main.py                      # Точка входа
├── config.py                    # Конфигурация
├── requirements.txt             # Зависимости
│
├── core/                        # Базовые компоненты
│   ├── logger.py               # Логирование
│   ├── exceptions.py           # Исключения
│   └── auth.py                 # Аутентификация
│
├── exchange/                    # Интеграция с биржей
│   ├── websocket_manager.py    # WebSocket менеджер
│   ├── rest_client.py          # REST API клиент
│   └── bybit_auth.py           # Аутентификация Bybit
│
├── models/                      # Модели данных
│   ├── user.py                 # Модели пользователя
│   ├── orderbook.py            # Модели стакана
│   ├── signal.py               # Модели сигналов
│   └── market_data.py          # Модели рынка
│
├── strategy/                    # Торговая логика
│   ├── orderbook_manager.py    # Управление стаканом
│   ├── analyzer.py             # Анализ данных
│   ├── strategy_engine.py      # Стратегия
│   └── risk_manager.py         # Риск-менеджмент
│
├── execution/                   # Исполнение ордеров
│   └── execution_manager.py    # Менеджер исполнения
│
├── api/                         # REST API
│   ├── app.py                  # FastAPI приложение
│   ├── routes.py               # Маршруты
│   └── websocket.py            # WebSocket для фронтенда
│
└── utils/                       # Утилиты
    ├── constants.py            # Константы
    └── helpers.py              # Вспомогательные функции
```

## 🔐 Безопасность

1. **JWT токены** - все API эндпоинты защищены JWT аутентификацией
2. **Хеширование паролей** - пароли хешируются с использованием bcrypt
3. **CORS** - настраиваемые CORS политики
4. **Переменные окружения** - чувствительные данные в .env файле

## 📝 Логирование

Логи сохраняются в директории `logs/`:
- `bot_YYYYMMDD.log` - общий лог
- `bot_errors_YYYYMMDD.log` - только ошибки

Уровни логирования настраиваются через `LOG_LEVEL` в `.env`