// frontend/src/types/orders.types.ts
/**
 * Типы данных для управления ордерами.
 */

/**
 * Тип ордера.
 */
export type OrderType = 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT';

/**
 * Сторона ордера.
 */
export type OrderSide = 'BUY' | 'SELL';

/**
 * Статус ордера.
 */
export type OrderStatus =
  | 'PENDING'           // Ожидает размещения
  | 'PLACED'            // Размещен на бирже
  | 'PARTIALLY_FILLED'  // Частично исполнен
  | 'FILLED'            // Полностью исполнен
  | 'CANCELLED'         // Отменен
  | 'REJECTED';         // Отклонен

/**
 * Базовая информация об ордере.
 */
export interface Order {
  // Идентификаторы
  order_id: string;              // UUID ордера в системе
  client_order_id: string;       // Client Order ID для биржи
  exchange_order_id?: string;    // ID ордера на бирже

  // Основные параметры
  symbol: string;                // Торговая пара
  side: OrderSide;               // Покупка/Продажа
  order_type: OrderType;         // Тип ордера

  // Количество и цена
  quantity: number;              // Количество
  price: number | null;          // Цена (null для MARKET)
  filled_quantity: number;       // Исполненное количество
  average_price: number;         // Средняя цена исполнения

  // Риск-менеджмент
  take_profit: number | null;    // Take Profit
  stop_loss: number | null;      // Stop Loss
  leverage: number;              // Плечо

  // Статус
  status: OrderStatus;

  // Временные метки
  created_at: string;            // ISO datetime создания
  updated_at: string;            // ISO datetime последнего обновления
  filled_at?: string;            // ISO datetime исполнения

  // Дополнительная информация
  strategy?: string;             // Название стратегии
  signal_id?: string;            // ID сигнала, создавшего ордер
  notes?: string;                // Примечания
}

/**
 * Расширенная информация об ордере для детального просмотра.
 */
export interface OrderDetail extends Order {
  // Текущий PnL
  current_pnl: number;           // Текущая прибыль/убыток в USDT
  current_pnl_percent: number;   // Текущая прибыль/убыток в %

  // Цены
  current_price: number;         // Текущая цена актива
  entry_price: number;           // Цена входа

  // Расчетные значения
  position_value: number;        // Стоимость позиции
  margin_used: number;           // Использованная маржа
  liquidation_price?: number;    // Цена ликвидации

  // Исполнение
  execution_time?: number;       // Время исполнения в мс
  fees: number;                  // Комиссии

  // История
  fills?: OrderFill[];           // История частичных исполнений
}

/**
 * Частичное исполнение ордера.
 */
export interface OrderFill {
  fill_id: string;
  quantity: number;
  price: number;
  fee: number;
  timestamp: string;
}

/**
 * Параметры для создания ордера.
 */
export interface CreateOrderParams {
  symbol: string;
  side: OrderSide;
  order_type: OrderType;
  quantity: number;
  price?: number;
  take_profit?: number;
  stop_loss?: number;
  leverage?: number;
  strategy?: string;
  signal_id?: string;
}

/**
 * Параметры для закрытия ордера.
 */
export interface CloseOrderParams {
  order_id: string;
  reason?: string; // Причина закрытия
}

/**
 * Ответ API при создании ордера.
 */
export interface CreateOrderResponse {
  success: boolean;
  order: Order;
  message?: string;
}

/**
 * Ответ API при закрытии ордера.
 */
export interface CloseOrderResponse {
  success: boolean;
  order_id: string;
  closed_at: string;
  final_pnl: number;
  message?: string;
}

/**
 * Фильтры для списка ордеров.
 */
export interface OrderFilters {
  symbol?: string;
  side?: OrderSide;
  status?: OrderStatus;
  strategy?: string;
  date_from?: string;
  date_to?: string;
}

/**
 * Статистика по ордерам.
 */
export interface OrdersStats {
  total: number;
  active: number;
  filled: number;
  cancelled: number;
  total_volume: number;
  total_pnl: number;
}