"""
Trailing Stop Manager - Python 3.9+ совместимая версия.

ИСПРАВЛЕНО:
- Заменен asyncio.timeout() на asyncio.wait_for() для Python 3.9/3.10
- Все timeout операции совместимы с Python 3.9+

Путь: backend/strategy/trailing_stop_manager.py
"""
import asyncio
from typing import Dict, Optional, List
from datetime import datetime

from backend.core.logger import get_logger
from backend.config import settings
from backend.models.signal import SignalType
from backend.database.models import PositionStatus, OrderSide
from backend.strategy.risk_models import TrailingStopState
from backend.infrastructure.repositories.position_repository import position_repository
from backend.exchange.rest_client import rest_client
from backend.core.trade_reporter import trade_reporter, PositionEvent, PositionEventType

logger = get_logger(__name__)


class TrailingStopManager:
    """
    Менеджер trailing stops для открытых позиций.

    PYTHON 3.9+ COMPATIBLE:
    - Использует asyncio.wait_for() вместо asyncio.timeout()
    """

    def __init__(self, trade_managers: Optional[Dict] = None):
        """
        Инициализация Trailing Stop Manager.

        Args:
            trade_managers: Dict[symbol, TradeManager] для market trades метрик
        """
        self.enabled = settings.TRAILING_STOP_ENABLED
        self.activation_profit_percent = settings.TRAILING_STOP_ACTIVATION_PROFIT_PERCENT / 100
        self.distance_percent = settings.TRAILING_STOP_DISTANCE_PERCENT / 100
        self.update_interval = settings.TRAILING_STOP_UPDATE_INTERVAL_SEC
        self.trade_managers = trade_managers or {}

        # Активные trailing stops: {symbol: TrailingStopState}
        self.active_trails: Dict[str, TrailingStopState] = {}

        # Задачи
        self.update_task: Optional[asyncio.Task] = None
        self.initialization_task: Optional[asyncio.Task] = None

        # Флаг инициализации
        self.is_initialized = False

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
            f"update_interval={self.update_interval}s, "
            f"trade_managers={len(self.trade_managers)} symbols"
        )

    async def start(self):
        """Запуск Trailing Stop Manager (NON-BLOCKING)."""
        if not self.enabled:
            logger.info("TrailingStopManager отключен в конфигурации")
            return

        logger.info("Запуск TrailingStopManager (non-blocking)...")

        try:
            # Фоновая инициализация
            self.initialization_task = asyncio.create_task(
                self._initialize_in_background()
            )

            # Немедленный запуск update loop
            self.update_task = asyncio.create_task(self._update_loop())

            logger.info(
                "✓ TrailingStopManager запущен (фоновая инициализация) | "
                f"Update loop активен"
            )

        except Exception as e:
            logger.error(
                f"Ошибка запуска TrailingStopManager: {e}",
                exc_info=True
            )
            logger.warning("TrailingStopManager продолжит работу с пустым состоянием")

    async def _initialize_in_background(self):
        """Фоновая инициализация - загрузка активных позиций."""
        logger.info("Фоновая загрузка активных позиций...")

        try:
            await asyncio.sleep(2)
            await self._load_active_positions()
            self.is_initialized = True

            logger.info(
                f"✓ TrailingStopManager инициализирован | "
                f"Загружено позиций: {len(self.active_trails)}"
            )

        except Exception as e:
            logger.error(
                f"Ошибка фоновой инициализации TrailingStopManager: {e}",
                exc_info=True
            )
            self.is_initialized = True
            logger.warning(
                "TrailingStopManager будет работать с пустым начальным состоянием"
            )

    async def stop(self):
        """Остановка Trailing Stop Manager."""
        logger.info("Остановка TrailingStopManager...")

        if self.initialization_task and not self.initialization_task.done():
            self.initialization_task.cancel()
            try:
                await self.initialization_task
            except asyncio.CancelledError:
                pass

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
        """Загрузка активных позиций из БД."""
        try:
            # ИСПРАВЛЕНО: asyncio.wait_for вместо asyncio.timeout
            positions = await asyncio.wait_for(
                position_repository.get_by_status(PositionStatus.OPEN),
                timeout=10.0
            )

            for position in positions:
                try:
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
                        f"SL=${position.stop_loss:.2f}" if position.stop_loss else f"SL=None"
                    )

                except Exception as e:
                    logger.error(
                        f"Ошибка загрузки trailing stop для {position.symbol}: {e}",
                        exc_info=True
                    )

            logger.info(
                f"Загружено {len(self.active_trails)} активных позиций "
                f"для trailing stop"
            )

        except asyncio.TimeoutError:
            logger.error(
                "Timeout при загрузке активных позиций (10s). "
                "Возможны проблемы с БД."
            )
        except Exception as e:
            logger.error(
                f"Ошибка загрузки активных позиций: {e}",
                exc_info=True
            )

    async def _update_loop(self):
        """Основной цикл обновления trailing stops."""
        logger.info("Цикл обновления trailing stops запущен")

        wait_count = 0
        while not self.is_initialized and wait_count < 30:
            logger.debug("Ожидание инициализации TrailingStopManager...")
            await asyncio.sleep(1)
            wait_count += 1

        if not self.is_initialized:
            logger.warning(
                "TrailingStopManager не инициализирован после 30 секунд, "
                "но update loop продолжит работу"
            )

        logger.info("Update loop TrailingStopManager активен")

        while True:
            try:
                if self.active_trails:
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
                # ИСПРАВЛЕНО: asyncio.wait_for вместо asyncio.timeout
                trail_state = self.active_trails[symbol]
                await asyncio.wait_for(
                    self._update_trailing_stop(symbol, trail_state),
                    timeout=5.0
                )

            except asyncio.TimeoutError:
                logger.warning(
                    f"{symbol} | Timeout обновления trailing stop (5s)"
                )
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
        """Обновление trailing stop для конкретной позиции."""
        # ===== ШАГ 1: ПРОВЕРКА СУЩЕСТВОВАНИЯ ПОЗИЦИИ =====
        try:
            # ИСПРАВЛЕНО: asyncio.wait_for вместо asyncio.timeout
            position = await asyncio.wait_for(
                position_repository.find_open_by_symbol(symbol),
                timeout=3.0
            )

        except asyncio.TimeoutError:
            logger.warning(f"{symbol} | Timeout получения позиции из БД")
            return
        except Exception as e:
            logger.error(f"{symbol} | Ошибка получения позиции: {e}")
            return

        if not position or position.status != PositionStatus.OPEN:
            if symbol in self.active_trails:
                del self.active_trails[symbol]
                logger.debug(f"{symbol} | Позиция закрыта, trailing stop удален")
            return

        # ===== ШАГ 2: ПОЛУЧЕНИЕ ТЕКУЩЕЙ ЦЕНЫ =====
        current_price = position.current_price or position.entry_price
        trail_state.current_price = current_price

        # ===== ШАГ 3: ОБНОВЛЕНИЕ HIGHEST/LOWEST PRICE =====
        side = position.side

        if side == OrderSide.BUY:
            if current_price > trail_state.highest_price:
                trail_state.highest_price = current_price
        else:
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

                # Логируем в trades.log
                trade_reporter.log_position_event(PositionEvent(
                    event_type=PositionEventType.TRAILING_ACTIVATED,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    position_id=str(position.id),
                    side=position.side.value,
                    entry_price=trail_state.entry_price,
                    current_price=current_price,
                    quantity=position.quantity,
                    unrealized_pnl_percent=profit_percent,
                    activation_profit_percent=trail_state.activation_profit_percent,
                    reason=f"Profit {profit_percent:.2%} >= activation {trail_state.activation_profit_percent:.2%}"
                ))
            else:
                return

        # ===== ШАГ 6: РАСЧЕТ НОВОГО STOP LOSS =====
        trail_distance = self._get_trail_distance(
            profit_percent=profit_percent,
            symbol=symbol,
            side=side
        )
        trail_state.trailing_distance_percent = trail_distance

        if side == OrderSide.BUY:
            new_stop_loss = trail_state.highest_price * (1 - trail_distance)
        else:
            new_stop_loss = trail_state.lowest_price * (1 + trail_distance)

        # ===== ШАГ 7: ПРОВЕРКА НЕОБХОДИМОСТИ ОБНОВЛЕНИЯ =====
        if side == OrderSide.BUY:
            should_update = new_stop_loss > trail_state.current_stop_loss
        else:
            should_update = new_stop_loss < trail_state.current_stop_loss

        if not should_update:
            return

        # ===== ШАГ 8: ОБНОВЛЕНИЕ STOP LOSS =====
        old_sl = trail_state.current_stop_loss
        trail_state.current_stop_loss = new_stop_loss
        trail_state.updated_at = datetime.now()

        # Обновляем в БД
        try:
            # ИСПРАВЛЕНО: asyncio.wait_for вместо asyncio.timeout
            await asyncio.wait_for(
                position_repository.update_stop_loss(
                    position_id=trail_state.position_id,
                    new_stop_loss=new_stop_loss
                ),
                timeout=3.0
            )
        except asyncio.TimeoutError:
            logger.error(f"{symbol} | Timeout обновления SL в БД")
            trail_state.current_stop_loss = old_sl
            return
        except Exception as e:
            logger.error(f"{symbol} | Ошибка обновления SL в БД: {e}")
            trail_state.current_stop_loss = old_sl
            return

        # Обновляем на бирже через REST API
        try:
            if not rest_client.session:
                logger.warning(
                    f"{symbol} | REST client не инициализирован, "
                    f"пропуск обновления на бирже"
                )
                return

            # ИСПРАВЛЕНО: asyncio.wait_for вместо asyncio.timeout
            await asyncio.wait_for(
                rest_client.set_trading_stop(
                    symbol=symbol,
                    stop_loss=new_stop_loss
                ),
                timeout=5.0
            )

            self.total_updates += 1

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

            # Логируем в trades.log
            # Получаем market stats если доступны
            arrival_rate = None
            buy_sell_ratio = None
            trade_manager = self.trade_managers.get(symbol)
            if trade_manager:
                try:
                    stats = trade_manager.get_statistics(window_seconds=60)
                    if stats:
                        arrival_rate = stats.trades_per_second
                        buy_sell_ratio = stats.buy_sell_ratio
                except Exception:
                    pass

            trade_reporter.log_position_event(PositionEvent(
                event_type=PositionEventType.TRAILING_UPDATED,
                symbol=symbol,
                timestamp=datetime.now(),
                position_id=str(position.id),
                side=position.side.value,
                entry_price=trail_state.entry_price,
                current_price=current_price,
                quantity=position.quantity,
                unrealized_pnl_percent=profit_percent,
                old_stop_loss=old_sl,
                new_stop_loss=new_stop_loss,
                trail_distance_percent=trail_distance,
                locked_profit=locked_profit,
                arrival_rate=arrival_rate,
                buy_sell_ratio=buy_sell_ratio,
                reason=f"Distance: {trail_distance:.2%}, Locked: ${locked_profit:.2f}"
            ))

        except asyncio.TimeoutError:
            logger.error(
                f"{symbol} | Timeout обновления SL на бирже (5s)"
            )
            trail_state.current_stop_loss = old_sl

        except Exception as e:
            logger.error(
                f"{symbol} | Ошибка обновления SL на бирже: {e}",
                exc_info=True
            )
            trail_state.current_stop_loss = old_sl

    def _get_trail_distance(
        self,
        profit_percent: float,
        symbol: str,
        side: OrderSide
    ) -> float:
        """
        Определение trail distance на основе текущей прибыли,
        arrival rate и buy/sell pressure.

        ЛОГИКА:
        1. Базовая дистанция из ступенчатых уровней
        2. Адаптация на основе arrival rate (высокая волатильность → шире SL)
        3. Адаптация на основе buy/sell pressure (давление против позиции → уже SL)

        Args:
            profit_percent: Текущая прибыль позиции
            symbol: Торговая пара
            side: Сторона позиции (BUY/SELL)

        Returns:
            Адаптированная дистанция trailing stop
        """
        # Базовая дистанция из ступенчатых уровней
        base_distance = self.distance_percent
        for level in reversed(self.distance_levels):
            if profit_percent >= level['profit']:
                base_distance = level['distance']
                break

        # Если нет TradeManager - возвращаем базовую дистанцию
        trade_manager = self.trade_managers.get(symbol)
        if not trade_manager:
            return base_distance

        try:
            # Получаем метрики за 60 секунд
            stats = trade_manager.get_statistics(window_seconds=60)
            if not stats:
                return base_distance

            arrival_rate = stats.trades_per_second
            buy_sell_ratio = stats.buy_sell_ratio

            # ===== АДАПТАЦИЯ 1: ARRIVAL RATE (ВОЛАТИЛЬНОСТЬ) =====
            # Высокий arrival rate → быстрый рынок → расширяем дистанцию
            # Низкий arrival rate → медленный рынок → сужаем дистанцию
            volatility_multiplier = 1.0

            if arrival_rate > 10.0:  # Очень быстрый рынок (> 10 trades/sec)
                volatility_multiplier = 1.3  # +30% к дистанции
            elif arrival_rate > 5.0:  # Быстрый рынок (5-10 trades/sec)
                volatility_multiplier = 1.15  # +15% к дистанции
            elif arrival_rate < 1.0:  # Медленный рынок (< 1 trade/sec)
                volatility_multiplier = 0.85  # -15% к дистанции

            # ===== АДАПТАЦИЯ 2: BUY/SELL PRESSURE =====
            # Если давление против нашей позиции → сужаем дистанцию (защита)
            pressure_multiplier = 1.0

            if side == OrderSide.BUY:
                # LONG позиция: опасно если sell pressure (ratio < 1.0)
                if buy_sell_ratio < 0.8:  # Сильное давление продаж
                    pressure_multiplier = 0.75  # -25% (подтягиваем SL ближе)
                elif buy_sell_ratio < 1.0:  # Умеренное давление продаж
                    pressure_multiplier = 0.9  # -10%
                elif buy_sell_ratio > 1.5:  # Сильное давление покупок
                    pressure_multiplier = 1.1  # +10% (даем больше пространства)

            else:  # SHORT позиция
                # SHORT позиция: опасно если buy pressure (ratio > 1.0)
                if buy_sell_ratio > 1.25:  # Сильное давление покупок
                    pressure_multiplier = 0.75  # -25% (подтягиваем SL ближе)
                elif buy_sell_ratio > 1.0:  # Умеренное давление покупок
                    pressure_multiplier = 0.9  # -10%
                elif buy_sell_ratio < 0.7:  # Сильное давление продаж
                    pressure_multiplier = 1.1  # +10% (даем больше пространства)

            # Итоговая дистанция
            adaptive_distance = base_distance * volatility_multiplier * pressure_multiplier

            # Ограничиваем диапазон (не меньше 0.3%, не больше 3%)
            adaptive_distance = max(0.003, min(adaptive_distance, 0.03))

            logger.debug(
                f"{symbol} | Trail distance adaptive calculation: "
                f"base={base_distance:.2%}, "
                f"arrival_rate={arrival_rate:.2f}t/s (x{volatility_multiplier:.2f}), "
                f"buy/sell={buy_sell_ratio:.2f} (x{pressure_multiplier:.2f}), "
                f"final={adaptive_distance:.2%}"
            )

            return adaptive_distance

        except Exception as e:
            logger.warning(
                f"{symbol} | Error in adaptive trail distance: {e}",
                exc_info=False
            )
            return base_distance

    def register_position_opened(
        self,
        symbol: str,
        position_id: str,
        entry_price: float,
        stop_loss: float,
        side: OrderSide
    ):
        """Регистрация новой позиции для trailing."""
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
        """Регистрация закрытия позиции."""
        if symbol in self.active_trails:
            del self.active_trails[symbol]
            logger.info(f"{symbol} | Trailing stop удален (позиция закрыта)")

    def update_position_price(self, symbol: str, current_price: float):
        """Обновление текущей цены позиции."""
        if symbol not in self.active_trails:
            return

        trail_state = self.active_trails[symbol]
        trail_state.current_price = current_price

    def get_trailing_status(self, symbol: str) -> Optional[Dict]:
        """Получение статуса trailing stop для символа."""
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
        """Получение всех активных trailing stops."""
        return {
            symbol: self.get_trailing_status(symbol)
            for symbol in self.active_trails
        }

    def get_statistics(self) -> Dict:
        """Получение статистики работы."""
        return {
            'enabled': self.enabled,
            'is_initialized': self.is_initialized,
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