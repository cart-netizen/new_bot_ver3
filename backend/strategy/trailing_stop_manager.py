"""
Trailing Stop Manager - Динамическое управление trailing stops.

ФУНКЦИОНАЛЬНОСТЬ:
- Activation trigger (начало trailing после достижения профита)
- Adaptive distance (на основе уровня прибыли)
- Multiple distance levels (ступенчатое увеличение)
- Auto-update через position FSM
- Интеграция с биржей через REST API

DISTANCE LEVELS:
- Profit 1.5-3%: Trail 0.8%
- Profit 3-5%: Trail 1.2%
- Profit 5%+: Trail 1.5%

Путь: backend/strategy/trailing_stop_manager.py
"""
import asyncio
from typing import Dict, Optional, List
from datetime import datetime

from core.logger import get_logger
from config import settings
from models.signal import SignalType
from database.models import PositionStatus, OrderSide
from strategy.risk_models import TrailingStopState
from infrastructure.repositories.position_repository import position_repository
from exchange.rest_client import rest_client

logger = get_logger(__name__)


class TrailingStopManager:
    """
    Менеджер trailing stops для открытых позиций.

    Автоматически отслеживает прибыльные позиции и подтягивает
    stop loss за ценой на заданном расстоянии.
    """

    def __init__(self):
        """Инициализация Trailing Stop Manager."""
        self.enabled = settings.TRAILING_STOP_ENABLED
        self.activation_profit_percent = settings.TRAILING_STOP_ACTIVATION_PROFIT_PERCENT / 100
        self.distance_percent = settings.TRAILING_STOP_DISTANCE_PERCENT / 100
        self.update_interval = settings.TRAILING_STOP_UPDATE_INTERVAL_SEC

        # Активные trailing stops: {symbol: TrailingStopState}
        self.active_trails: Dict[str, TrailingStopState] = {}

        # Задача обновления
        self.update_task: Optional[asyncio.Task] = None

        # Ступенчатые уровни дистанции
        self.distance_levels = [
            {'profit': 0.015, 'distance': 0.008},  # 1.5% profit → 0.8% trail
            {'profit': 0.030, 'distance': 0.012},  # 3% profit → 1.2% trail
            {'profit': 0.050, 'distance': 0.015},  # 5% profit → 1.5% trail
        ]

        # Метрики
        self.total_activations = 0
        self.total_updates = 0
        self.total_profits_locked = 0.0

        logger.info(
            f"TrailingStopManager инициализирован: "
            f"enabled={self.enabled}, "
            f"activation={self.activation_profit_percent:.1%}, "
            f"base_distance={self.distance_percent:.1%}, "
            f"update_interval={self.update_interval}s"
        )

    async def start(self):
        """Запуск Trailing Stop Manager."""
        if not self.enabled:
            logger.info("TrailingStopManager отключен в конфигурации")
            return

        logger.info("Запуск TrailingStopManager...")

        # Загружаем активные позиции из БД
        await self._load_active_positions()

        # Запускаем фоновую задачу обновления
        self.update_task = asyncio.create_task(self._update_loop())

        logger.info(
            f"✓ TrailingStopManager запущен | "
            f"Активных позиций: {len(self.active_trails)}"
        )

    async def stop(self):
        """Остановка Trailing Stop Manager."""
        logger.info("Остановка TrailingStopManager...")

        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"✓ TrailingStopManager остановлен | "
            f"Статистика: "
            f"activations={self.total_activations}, "
            f"updates={self.total_updates}, "
            f"locked_profit=${self.total_profits_locked:.2f}"
        )

    async def _load_active_positions(self):
        """Загрузка активных позиций из БД для инициализации."""
        try:
            positions = await position_repository.get_by_status(PositionStatus.OPEN)

            for position in positions:
                # Создаем TrailingStopState для каждой позиции
                state = TrailingStopState(
                    position_id=str(position.id),
                    symbol=position.symbol,
                    entry_price=position.entry_price,
                    current_price=position.current_price or position.entry_price,
                    highest_price=position.current_price or position.entry_price,
                    lowest_price=position.current_price or position.entry_price,
                    current_stop_loss=position.stop_loss,
                    trailing_distance_percent=self.distance_percent,
                    activation_profit_percent=self.activation_profit_percent,
                    is_active=False,
                    updated_at=datetime.now()
                )

                self.active_trails[position.symbol] = state

                logger.debug(
                    f"{position.symbol} | Загружен trailing stop: "
                    f"entry=${position.entry_price:.2f}, "
                    f"SL=${position.stop_loss:.2f}"
                )

            logger.info(
                f"Загружено {len(self.active_trails)} активных позиций "
                f"для trailing stop"
            )

        except Exception as e:
            logger.error(f"Ошибка загрузки активных позиций: {e}", exc_info=True)

    async def _update_loop(self):
        """Основной цикл обновления trailing stops."""
        logger.info("Цикл обновления trailing stops запущен")

        while True:
            try:
                await self._update_all_trailing_stops()
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                logger.info("Цикл обновления остановлен")
                break
            except Exception as e:
                logger.error(
                    f"Ошибка в цикле обновления trailing stops: {e}",
                    exc_info=True
                )
                await asyncio.sleep(self.update_interval)

    async def _update_all_trailing_stops(self):
        """Обновление всех активных trailing stops."""
        if not self.active_trails:
            return

        for symbol in list(self.active_trails.keys()):
            try:
                trail_state = self.active_trails[symbol]
                await self._update_trailing_stop(symbol, trail_state)

            except Exception as e:
                logger.error(
                    f"{symbol} | Ошибка обновления trailing stop: {e}",
                    exc_info=True
                )

    async def _update_trailing_stop(
        self,
        symbol: str,
        trail_state: TrailingStopState
    ):
        """
        Обновление trailing stop для конкретной позиции.

        Args:
            symbol: Торговая пара
            trail_state: Состояние trailing stop
        """
        # ===== ШАГ 1: ПРОВЕРКА СУЩЕСТВОВАНИЯ ПОЗИЦИИ =====
        position = await position_repository.find_open_by_symbol(symbol)

        if not position or position.status != PositionStatus.OPEN:
            # Позиция закрыта - удаляем trailing
            del self.active_trails[symbol]
            logger.debug(f"{symbol} | Позиция закрыта, trailing stop удален")
            return

        # ===== ШАГ 2: ПОЛУЧЕНИЕ ТЕКУЩЕЙ ЦЕНЫ =====
        current_price = position.current_price or position.entry_price

        # Обновляем текущую цену в state
        trail_state.current_price = current_price

        # ===== ШАГ 3: ОБНОВЛЕНИЕ HIGHEST/LOWEST PRICE =====
        side = position.side

        if side == OrderSide.BUY:
            # Для LONG позиции отслеживаем максимальную цену
            if current_price > trail_state.highest_price:
                trail_state.highest_price = current_price
        else:  # OrderSide.SELL
            # Для SHORT позиции отслеживаем минимальную цену
            if current_price < trail_state.lowest_price:
                trail_state.lowest_price = current_price

        # ===== ШАГ 4: РАСЧЕТ ТЕКУЩЕЙ ПРИБЫЛИ =====
        if side == OrderSide.BUY:
            profit_percent = (current_price - trail_state.entry_price) / trail_state.entry_price
        else:
            profit_percent = (trail_state.entry_price - current_price) / trail_state.entry_price

        # ===== ШАГ 5: ПРОВЕРКА АКТИВАЦИИ TRAILING =====
        if not trail_state.is_active:
            if profit_percent >= trail_state.activation_profit_percent:
                trail_state.is_active = True
                self.total_activations += 1

                logger.info(
                    f"{symbol} | 🎯 TRAILING STOP АКТИВИРОВАН | "
                    f"Прибыль: {profit_percent:.2%}, "
                    f"Порог активации: {trail_state.activation_profit_percent:.2%}"
                )
            else:
                # Еще не достигнут порог активации
                return

        # ===== ШАГ 6: РАСЧЕТ НОВОГО STOP LOSS =====

        # Определяем дистанцию на основе уровня прибыли
        trail_distance = self._get_trail_distance(profit_percent)
        trail_state.trailing_distance_percent = trail_distance

        # Рассчитываем новый stop loss
        if side == OrderSide.BUY:
            new_stop_loss = trail_state.highest_price * (1 - trail_distance)
        else:  # SHORT
            new_stop_loss = trail_state.lowest_price * (1 + trail_distance)

        # ===== ШАГ 7: ПРОВЕРКА НЕОБХОДИМОСТИ ОБНОВЛЕНИЯ =====
        if side == OrderSide.BUY:
            should_update = new_stop_loss > trail_state.current_stop_loss
        else:
            should_update = new_stop_loss < trail_state.current_stop_loss

        if not should_update:
            # Обновление не требуется
            return

        # ===== ШАГ 8: ОБНОВЛЕНИЕ STOP LOSS =====
        old_sl = trail_state.current_stop_loss
        trail_state.current_stop_loss = new_stop_loss
        trail_state.updated_at = datetime.now()

        # Обновляем в БД
        await position_repository.update_stop_loss(
            position_id=trail_state.position_id,
            new_stop_loss=new_stop_loss
        )

        # Обновляем на бирже через REST API
        try:
            await rest_client.set_trading_stop(
                symbol=symbol,
                stop_loss=new_stop_loss
            )

            self.total_updates += 1

            # Рассчитываем заблокированную прибыль
            if side == OrderSide.BUY:
                locked_profit = (new_stop_loss - trail_state.entry_price) * position.quantity
            else:
                locked_profit = (trail_state.entry_price - new_stop_loss) * position.quantity

            self.total_profits_locked += locked_profit

            logger.info(
                f"{symbol} | 📈 TRAILING STOP ОБНОВЛЕН | "
                f"Прибыль: {profit_percent:.2%}, "
                f"Дистанция: {trail_distance:.2%}, "
                f"SL: ${old_sl:.2f} → ${new_stop_loss:.2f} "
                f"(Δ{((new_stop_loss - old_sl) / old_sl):.2%}), "
                f"Заблокирована прибыль: ${locked_profit:.2f}"
            )

        except Exception as e:
            logger.error(
                f"{symbol} | Ошибка обновления SL на бирже: {e}",
                exc_info=True
            )
            # Откатываем изменения в state
            trail_state.current_stop_loss = old_sl

    def _get_trail_distance(self, profit_percent: float) -> float:
        """
        Определение trail distance на основе текущей прибыли.

        Ступенчатая система:
        - 1.5-3% profit: 0.8% trail
        - 3-5% profit: 1.2% trail
        - 5%+ profit: 1.5% trail

        Args:
            profit_percent: Текущая прибыль в долях (0.015 = 1.5%)

        Returns:
            float: Дистанция trailing в долях
        """
        for level in reversed(self.distance_levels):
            if profit_percent >= level['profit']:
                return level['distance']

        # Default - базовая дистанция
        return self.distance_percent

    def register_position_opened(
        self,
        symbol: str,
        position_id: str,
        entry_price: float,
        stop_loss: float,
        side: OrderSide
    ):
        """
        Регистрация новой позиции для trailing.

        Args:
            symbol: Торговая пара
            position_id: ID позиции
            entry_price: Цена входа
            stop_loss: Начальный stop loss
            side: Направление позиции
        """
        if not self.enabled:
            return

        state = TrailingStopState(
            position_id=position_id,
            symbol=symbol,
            entry_price=entry_price,
            current_price=entry_price,
            highest_price=entry_price,
            lowest_price=entry_price,
            current_stop_loss=stop_loss,
            trailing_distance_percent=self.distance_percent,
            activation_profit_percent=self.activation_profit_percent,
            is_active=False,
            updated_at=datetime.now()
        )

        self.active_trails[symbol] = state

        logger.info(
            f"{symbol} | Trailing stop зарегистрирован: "
            f"entry=${entry_price:.2f}, "
            f"initial_SL=${stop_loss:.2f}, "
            f"activation={self.activation_profit_percent:.2%}"
        )

    def register_position_closed(self, symbol: str):
        """
        Регистрация закрытия позиции.

        Args:
            symbol: Торговая пара
        """
        if symbol in self.active_trails:
            del self.active_trails[symbol]
            logger.info(f"{symbol} | Trailing stop удален (позиция закрыта)")

    def update_position_price(
        self,
        symbol: str,
        current_price: float
    ):
        """
        Обновление текущей цены позиции.

        Вызывается Position Monitor для обновления цены в real-time.

        Args:
            symbol: Торговая пара
            current_price: Текущая цена
        """
        if symbol not in self.active_trails:
            return

        trail_state = self.active_trails[symbol]
        trail_state.current_price = current_price

    def get_trailing_status(self, symbol: str) -> Optional[Dict]:
        """
        Получение статуса trailing stop для символа.

        Args:
            symbol: Торговая пара

        Returns:
            Dict со статусом или None если не найден
        """
        if symbol not in self.active_trails:
            return None

        state = self.active_trails[symbol]

        return {
            'symbol': symbol,
            'is_active': state.is_active,
            'entry_price': state.entry_price,
            'current_price': state.current_price,
            'highest_price': state.highest_price,
            'lowest_price': state.lowest_price,
            'current_stop_loss': state.current_stop_loss,
            'trailing_distance': state.trailing_distance_percent,
            'activation_profit': state.activation_profit_percent,
            'last_update': state.updated_at.isoformat()
        }

    def get_all_trailing_stops(self) -> Dict[str, Dict]:
        """
        Получение всех активных trailing stops.

        Returns:
            Dict[symbol, status]: Словарь со всеми trailing stops
        """
        return {
            symbol: self.get_trailing_status(symbol)
            for symbol in self.active_trails
        }

    def get_statistics(self) -> Dict:
        """
        Получение статистики работы Trailing Stop Manager.

        Returns:
            Dict: Статистика
        """
        return {
            'enabled': self.enabled,
            'active_positions': len(self.active_trails),
            'total_activations': self.total_activations,
            'total_updates': self.total_updates,
            'total_profits_locked': self.total_profits_locked,
            'activation_profit_threshold': self.activation_profit_percent,
            'base_distance': self.distance_percent,
            'update_interval': self.update_interval,
            'distance_levels': self.distance_levels
        }


# Глобальный экземпляр
trailing_stop_manager = TrailingStopManager()