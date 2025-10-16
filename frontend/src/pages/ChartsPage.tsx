// frontend/src/pages/ChartsPage.tsx

/**
 * Страница графиков.
 * Отображает графики выбранных торговых пар (по 3 в ряд).
 * Таймфрейм: 5 минут.
 * Обновление: каждые 15 секунд.
 */
export function ChartsPage() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Графики</h1>

      <div className="bg-surface border border-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-400 text-lg">
          Страница графиков будет реализована на следующем этапе
        </p>
        <p className="text-gray-500 text-sm mt-2">
          Здесь будут отображаться графики выбранных торговых пар по 3 в ряд
        </p>
      </div>
    </div>
  );
}