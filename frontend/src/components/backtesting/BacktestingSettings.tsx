// frontend/src/components/backtesting/BacktestingSettings.tsx

import { useState } from 'react';
import {
  Settings,
  TrendingUp,
  Database,
  Brain,
  Zap,
  ChevronDown,
  ChevronRight,
  DollarSign,
  Calendar,
  Target,
  Users,
  Activity,
  Save,
  Upload,
  RotateCcw
} from 'lucide-react';
import { Card } from '../ui/Card';
import { cn } from '../../utils/helpers';
import type { BacktestConfig } from '../../api/backtesting.api';

interface BacktestingSettingsProps {
  config: Partial<BacktestConfig>;
  onChange: (config: Partial<BacktestConfig>) => void;
  onSave?: () => void;
}

// Collapsible Section Component
interface SectionProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

function Section({ title, icon, children, defaultOpen = false }: SectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-4 bg-gray-800/30 hover:bg-gray-800/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          {icon}
          <h3 className="text-lg font-semibold text-white">{title}</h3>
        </div>
        {isOpen ? (
          <ChevronDown className="h-5 w-5 text-gray-400" />
        ) : (
          <ChevronRight className="h-5 w-5 text-gray-400" />
        )}
      </button>
      {isOpen && <div className="p-4 space-y-4">{children}</div>}
    </div>
  );
}

// Input Field Components
interface InputFieldProps {
  label: string;
  value: string | number;
  onChange: (value: string) => void;
  type?: 'text' | 'number' | 'date' | 'datetime-local';
  placeholder?: string;
  hint?: string;
  min?: number;
  max?: number;
  step?: number;
}

function InputField({ label, value, onChange, type = 'text', placeholder, hint, min, max, step }: InputFieldProps) {
  return (
    <div className="space-y-1.5">
      <label className="block text-sm font-medium text-gray-300">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        min={min}
        max={max}
        step={step}
        className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      />
      {hint && <p className="text-xs text-gray-500">{hint}</p>}
    </div>
  );
}

interface SelectFieldProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  hint?: string;
}

function SelectField({ label, value, onChange, options, hint }: SelectFieldProps) {
  return (
    <div className="space-y-1.5">
      <label className="block text-sm font-medium text-gray-300">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
      {hint && <p className="text-xs text-gray-500">{hint}</p>}
    </div>
  );
}

interface CheckboxFieldProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  hint?: string;
}

function CheckboxField({ label, checked, onChange, hint }: CheckboxFieldProps) {
  return (
    <div className="flex items-start space-x-3">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1 h-4 w-4 rounded border-gray-700 bg-gray-900 text-blue-500 focus:ring-2 focus:ring-blue-500 focus:ring-offset-0"
      />
      <div className="flex-1">
        <label className="block text-sm font-medium text-gray-300 cursor-pointer">
          {label}
        </label>
        {hint && <p className="text-xs text-gray-500 mt-0.5">{hint}</p>}
      </div>
    </div>
  );
}

export function BacktestingSettings({ config, onChange, onSave }: BacktestingSettingsProps) {
  const updateConfig = (updates: Partial<BacktestConfig>) => {
    onChange({ ...config, ...updates });
  };

  const handleReset = () => {
    // Reset to default values
    const defaults: Partial<BacktestConfig> = {
      symbol: 'BTCUSDT',
      candle_interval: '1m',
      initial_capital: 10000,
      commission_rate: 0.001,
      slippage_pct: 0.1,
      slippage_model: 'fixed',
      simulate_latency: false,
      consensus_mode: 'weighted',
      min_strategies_for_signal: 2,
      min_consensus_confidence: 0.5,
      position_size_pct: 10,
      position_size_mode: 'fixed_percent',
      max_open_positions: 3,
      stop_loss_pct: 2,
      take_profit_pct: 4,
      use_trailing_stop: false,
      trailing_stop_activation_pct: 2,
      trailing_stop_distance_pct: 1,
      risk_per_trade_pct: 1,
      use_orderbook_data: false,
      orderbook_num_levels: 20,
      orderbook_base_spread_bps: 2.0,
      use_market_trades: false,
      trades_per_volume_unit: 100,
      use_ml_model: false,
      ml_server_url: 'http://localhost:8001',
      use_cache: true,
      warmup_period_bars: 100,
      verbose: false,
      log_trades: false,
    };
    onChange(defaults);
  };

  return (
    <div className="space-y-4">
      {/* Header Actions */}
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Settings className="h-6 w-6 text-blue-400" />
            <h2 className="text-xl font-bold text-white">Настройки бэктестинга</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleReset}
              className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg transition-colors"
            >
              <RotateCcw className="h-4 w-4" />
              Сбросить
            </button>
            {onSave && (
              <button
                onClick={onSave}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
              >
                <Save className="h-4 w-4" />
                Сохранить шаблон
              </button>
            )}
          </div>
        </div>
      </Card>

      {/* 1. Основные настройки */}
      <Section
        title="Основные настройки"
        icon={<Calendar className="h-5 w-5 text-blue-400" />}
        defaultOpen={true}
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InputField
            label="Название теста"
            value={config.name || ''}
            onChange={(v) => updateConfig({ name: v })}
            placeholder="Мой бэктест"
            hint="Название для идентификации теста"
          />
          <InputField
            label="Торговая пара"
            value={config.symbol || 'BTCUSDT'}
            onChange={(v) => updateConfig({ symbol: v })}
            placeholder="BTCUSDT"
            hint="Символ торговой пары"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <InputField
            label="Дата начала"
            value={config.start_date || ''}
            onChange={(v) => updateConfig({ start_date: v })}
            type="datetime-local"
            hint="Начало периода тестирования"
          />
          <InputField
            label="Дата окончания"
            value={config.end_date || ''}
            onChange={(v) => updateConfig({ end_date: v })}
            type="datetime-local"
            hint="Конец периода тестирования"
          />
          <SelectField
            label="Интервал свечей"
            value={config.candle_interval || '1m'}
            onChange={(v) => updateConfig({ candle_interval: v })}
            options={[
              { value: '1m', label: '1 минута' },
              { value: '5m', label: '5 минут' },
              { value: '15m', label: '15 минут' },
              { value: '1h', label: '1 час' },
              { value: '4h', label: '4 часа' },
              { value: '1d', label: '1 день' },
            ]}
            hint="Таймфрейм свечей"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InputField
            label="Начальный капитал (USDT)"
            value={config.initial_capital || 10000}
            onChange={(v) => updateConfig({ initial_capital: parseFloat(v) })}
            type="number"
            min={100}
            step={100}
            hint="Стартовый капитал для теста"
          />
          <InputField
            label="Warmup Period (свечей)"
            value={config.warmup_period_bars || 100}
            onChange={(v) => updateConfig({ warmup_period_bars: parseInt(v) })}
            type="number"
            min={0}
            hint="Количество свечей для прогрева индикаторов"
          />
        </div>

        <InputField
          label="Описание"
          value={config.description || ''}
          onChange={(v) => updateConfig({ description: v })}
          placeholder="Описание стратегии и условий теста"
          hint="Опциональное описание теста"
        />
      </Section>

      {/* 2. Управление рисками */}
      <Section
        title="Управление рисками"
        icon={<Target className="h-5 w-5 text-red-400" />}
        defaultOpen={true}
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InputField
            label="Размер позиции (%)"
            value={config.position_size_pct || 10}
            onChange={(v) => updateConfig({ position_size_pct: parseFloat(v) })}
            type="number"
            min={0.1}
            max={100}
            step={0.1}
            hint="Процент от капитала на одну сделку"
          />
          <SelectField
            label="Режим расчета позиции"
            value={config.position_size_mode || 'fixed_percent'}
            onChange={(v) => updateConfig({ position_size_mode: v })}
            options={[
              { value: 'fixed_percent', label: 'Фиксированный процент' },
              { value: 'risk_based', label: 'На основе риска' },
              { value: 'volatility_adjusted', label: 'С учетом волатильности' },
            ]}
            hint="Метод расчета размера позиции"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <InputField
            label="Stop Loss (%)"
            value={config.stop_loss_pct || 2}
            onChange={(v) => updateConfig({ stop_loss_pct: parseFloat(v) })}
            type="number"
            min={0.1}
            max={50}
            step={0.1}
            hint="Процент убытка для закрытия позиции"
          />
          <InputField
            label="Take Profit (%)"
            value={config.take_profit_pct || 4}
            onChange={(v) => updateConfig({ take_profit_pct: parseFloat(v) })}
            type="number"
            min={0.1}
            max={100}
            step={0.1}
            hint="Процент прибыли для закрытия позиции"
          />
          <InputField
            label="Макс. открытых позиций"
            value={config.max_open_positions || 3}
            onChange={(v) => updateConfig({ max_open_positions: parseInt(v) })}
            type="number"
            min={1}
            max={10}
            hint="Максимум одновременных сделок"
          />
        </div>

        <CheckboxField
          label="Использовать трейлинг-стоп"
          checked={config.use_trailing_stop || false}
          onChange={(v) => updateConfig({ use_trailing_stop: v })}
          hint="Автоматическое подтягивание стоп-лосса за ценой"
        />

        {config.use_trailing_stop && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 ml-7">
            <InputField
              label="Активация трейлинга (%)"
              value={config.trailing_stop_activation_pct || 2}
              onChange={(v) => updateConfig({ trailing_stop_activation_pct: parseFloat(v) })}
              type="number"
              min={0.1}
              step={0.1}
              hint="Профит для активации трейлинга"
            />
            <InputField
              label="Дистанция трейлинга (%)"
              value={config.trailing_stop_distance_pct || 1}
              onChange={(v) => updateConfig({ trailing_stop_distance_pct: parseFloat(v) })}
              type="number"
              min={0.1}
              step={0.1}
              hint="Расстояние стопа от цены"
            />
          </div>
        )}

        <InputField
          label="Риск на сделку (%)"
          value={config.risk_per_trade_pct || 1}
          onChange={(v) => updateConfig({ risk_per_trade_pct: parseFloat(v) })}
          type="number"
          min={0.1}
          max={10}
          step={0.1}
          hint="Максимальный процент риска от капитала на сделку"
        />
      </Section>

      {/* 3. Стратегии и консенсус */}
      <Section
        title="Стратегии и консенсус"
        icon={<Users className="h-5 w-5 text-purple-400" />}
        defaultOpen={true}
      >
        <SelectField
          label="Режим консенсуса"
          value={config.consensus_mode || 'weighted'}
          onChange={(v) => updateConfig({ consensus_mode: v })}
          options={[
            { value: 'weighted', label: 'Взвешенный (Weighted)' },
            { value: 'majority', label: 'Большинство (Majority)' },
            { value: 'unanimous', label: 'Единогласный (Unanimous)' },
          ]}
          hint="Метод принятия решения на основе сигналов стратегий"
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InputField
            label="Мин. стратегий для сигнала"
            value={config.min_strategies_for_signal || 2}
            onChange={(v) => updateConfig({ min_strategies_for_signal: parseInt(v) })}
            type="number"
            min={1}
            max={10}
            hint="Минимальное количество стратегий для открытия позиции"
          />
          <InputField
            label="Мин. уверенность консенсуса"
            value={config.min_consensus_confidence || 0.5}
            onChange={(v) => updateConfig({ min_consensus_confidence: parseFloat(v) })}
            type="number"
            min={0}
            max={1}
            step={0.05}
            hint="Минимальная уверенность для открытия позиции (0-1)"
          />
        </div>

        {/* TODO: Enabled strategies multi-select */}
        <div className="p-3 bg-gray-800/30 rounded-lg border border-gray-700">
          <p className="text-sm text-gray-400">
            Выбор стратегий будет доступен после подключения к бэкенду
          </p>
        </div>
      </Section>

      {/* 4. OrderBook и Market Trades (Фаза 1) */}
      <Section
        title="OrderBook и Market Trades"
        icon={<Activity className="h-5 w-5 text-green-400" />}
      >
        <CheckboxField
          label="Использовать данные OrderBook"
          checked={config.use_orderbook_data || false}
          onChange={(v) => updateConfig({ use_orderbook_data: v })}
          hint="Генерация и использование синтетического стакана ордеров"
        />

        {config.use_orderbook_data && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 ml-7">
            <InputField
              label="Количество уровней"
              value={config.orderbook_num_levels || 20}
              onChange={(v) => updateConfig({ orderbook_num_levels: parseInt(v) })}
              type="number"
              min={10}
              max={50}
              hint="Глубина стакана (10-50)"
            />
            <InputField
              label="Базовый спред (bps)"
              value={config.orderbook_base_spread_bps || 2.0}
              onChange={(v) => updateConfig({ orderbook_base_spread_bps: parseFloat(v) })}
              type="number"
              min={0.1}
              max={50}
              step={0.1}
              hint="Базовый спред в basis points"
            />
          </div>
        )}

        <CheckboxField
          label="Использовать Market Trades"
          checked={config.use_market_trades || false}
          onChange={(v) => updateConfig({ use_market_trades: v })}
          hint="Генерация синтетических рыночных сделок"
        />

        {config.use_market_trades && (
          <div className="ml-7">
            <InputField
              label="Сделок на единицу объема"
              value={config.trades_per_volume_unit || 100}
              onChange={(v) => updateConfig({ trades_per_volume_unit: parseInt(v) })}
              type="number"
              min={10}
              max={1000}
              hint="Количество генерируемых сделок на единицу объема свечи"
            />
          </div>
        )}
      </Section>

      {/* 5. ML Model Integration (Фаза 2) */}
      <Section
        title="Интеграция ML модели"
        icon={<Brain className="h-5 w-5 text-pink-400" />}
      >
        <CheckboxField
          label="Использовать ML модель"
          checked={config.use_ml_model || false}
          onChange={(v) => updateConfig({ use_ml_model: v })}
          hint="Подключение внешней ML модели для предсказаний"
        />

        {config.use_ml_model && (
          <div className="space-y-4 ml-7">
            <InputField
              label="URL ML сервера"
              value={config.ml_server_url || 'http://localhost:8001'}
              onChange={(v) => updateConfig({ ml_server_url: v })}
              placeholder="http://localhost:8001"
              hint="Адрес сервера с ML моделью"
            />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <InputField
                label="Название модели"
                value={config.ml_model_name || ''}
                onChange={(v) => updateConfig({ ml_model_name: v })}
                placeholder="trading_model_v1"
                hint="Опционально: имя модели"
              />
              <InputField
                label="Версия модели"
                value={config.ml_model_version || ''}
                onChange={(v) => updateConfig({ ml_model_version: v })}
                placeholder="1.0.0"
                hint="Опционально: версия модели"
              />
            </div>
          </div>
        )}
      </Section>

      {/* 6. Оптимизация производительности (Фаза 3) */}
      <Section
        title="Оптимизация производительности"
        icon={<Zap className="h-5 w-5 text-yellow-400" />}
      >
        <CheckboxField
          label="Использовать кэширование"
          checked={config.use_cache !== false}
          onChange={(v) => updateConfig({ use_cache: v })}
          hint="Кэширование исторических данных для ускорения"
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InputField
            label="Генерировать OrderBook каждые N свечей"
            value={config.skip_orderbook_generation_every_n || 0}
            onChange={(v) => updateConfig({ skip_orderbook_generation_every_n: parseInt(v) || undefined })}
            type="number"
            min={0}
            hint="0 = генерировать всегда, N > 0 = пропускать генерацию"
          />
          <InputField
            label="Генерировать Trades каждые N свечей"
            value={config.skip_trades_generation_every_n || 0}
            onChange={(v) => updateConfig({ skip_trades_generation_every_n: parseInt(v) || undefined })}
            type="number"
            min={0}
            hint="0 = генерировать всегда, N > 0 = пропускать генерацию"
          />
        </div>
      </Section>

      {/* 7. Настройки биржи */}
      <Section
        title="Настройки биржи"
        icon={<DollarSign className="h-5 w-5 text-green-400" />}
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InputField
            label="Комиссия (%)"
            value={config.commission_rate || 0.1}
            onChange={(v) => updateConfig({ commission_rate: parseFloat(v) })}
            type="number"
            min={0}
            max={1}
            step={0.01}
            hint="Комиссия биржи в процентах"
          />
          <InputField
            label="Проскальзывание (%)"
            value={config.slippage_pct || 0.1}
            onChange={(v) => updateConfig({ slippage_pct: parseFloat(v) })}
            type="number"
            min={0}
            max={5}
            step={0.01}
            hint="Процент проскальзывания при исполнении"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <SelectField
            label="Модель проскальзывания"
            value={config.slippage_model || 'fixed'}
            onChange={(v) => updateConfig({ slippage_model: v as any })}
            options={[
              { value: 'fixed', label: 'Фиксированное' },
              { value: 'volume_based', label: 'На основе объема' },
              { value: 'percentage', label: 'Процентное' },
            ]}
            hint="Метод расчета проскальзывания"
          />
          <div className="flex items-center pt-7">
            <CheckboxField
              label="Симуляция задержки"
              checked={config.simulate_latency || false}
              onChange={(v) => updateConfig({ simulate_latency: v })}
              hint="Имитация задержки исполнения ордеров"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InputField
            label="Maker комиссия (%)"
            value={config.maker_commission || 0}
            onChange={(v) => updateConfig({ maker_commission: parseFloat(v) || undefined })}
            type="number"
            min={0}
            max={1}
            step={0.01}
            hint="Опционально: комиссия maker ордеров"
          />
          <InputField
            label="Taker комиссия (%)"
            value={config.taker_commission || 0}
            onChange={(v) => updateConfig({ taker_commission: parseFloat(v) || undefined })}
            type="number"
            min={0}
            max={1}
            step={0.01}
            hint="Опционально: комиссия taker ордеров"
          />
        </div>
      </Section>

      {/* 8. Отладка */}
      <Section
        title="Отладка"
        icon={<Database className="h-5 w-5 text-gray-400" />}
      >
        <CheckboxField
          label="Подробные логи"
          checked={config.verbose || false}
          onChange={(v) => updateConfig({ verbose: v })}
          hint="Вывод детальной информации о процессе бэктеста"
        />
        <CheckboxField
          label="Логировать сделки"
          checked={config.log_trades || false}
          onChange={(v) => updateConfig({ log_trades: v })}
          hint="Записывать информацию о каждой сделке в логи"
        />
      </Section>
    </div>
  );
}
