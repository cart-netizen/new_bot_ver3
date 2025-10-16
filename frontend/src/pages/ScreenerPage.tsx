// frontend/src/pages/ScreenerPage.tsx

/**
 * Страница скринера.
 * Отображает динамику изменения цены по каждому таймфрейму.
 * Фильтрация: объем > 4,000,000 USDT за 24 часа.
 */
export function ScreenerPage() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Скринер</h1>

      <div className="bg-surface border border-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-400 text-lg">
          Страница скринера будет реализована на следующем этапе
        </p>
        <p className="text-gray-500 text-sm mt-2">
          Здесь будет отображаться динамика изменения цены по таймфреймам (5m, 15m, 1h, 4h)
        </p>
      </div>
    </div>
  );
}