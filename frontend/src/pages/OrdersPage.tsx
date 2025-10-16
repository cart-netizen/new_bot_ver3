
// frontend/src/pages/OrdersPage.tsx

/**
 * Страница ордеров.
 * Отображает открытые ордера с краткой информацией.
 * При клике на ордер открывается детальная информация с графиком и кнопкой закрытия.
 */
export function OrdersPage() {
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Ордера</h1>

      <div className="bg-surface border border-gray-800 rounded-lg p-8 text-center">
        <p className="text-gray-400 text-lg">
          Страница ордеров будет реализована на следующем этапе
        </p>
        <p className="text-gray-500 text-sm mt-2">
          Здесь будут отображаться открытые ордера с возможностью детального просмотра
        </p>
      </div>
    </div>
  );
}