# Обновление фронтенда ML Management - Добавление V2 параметров

## Файл для обновления

`frontend/src/pages/MLManagementPage.tsx`

---

## Изменения

### 1. ✅ Interface TrainingParams - ОБНОВЛЕН

### 2. ✅ useState trainingParams - ОБНОВЛЕН

### 3. Добавить новые input поля в renderTrainingTab()

**Местоположение:** После строки 877 (после поля "Early Stopping Patience"), перед секцией "Options"

**Добавить следующий код:**

```tsx
          {/* Weight Decay */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Weight Decay (L2 Regularization)
              <Tooltip content="Регуляризация L2. Контролирует переобучение. Рекомендуется: 0.01. Больше = сильнее регуляризация.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.001"
              min="0"
              max="1"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.weight_decay}
              onChange={e =>
                setTrainingParams({ ...trainingParams, weight_decay: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2 рекомендуется: 0.01 (было: ~0 в v1)
            </p>
          </div>

          {/* Dropout */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Dropout
              <Tooltip content="Вероятность отключения нейронов. Предотвращает переобучение. Рекомендуется: 0.4. Выше = сильнее регуляризация.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="0.9"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.dropout}
              onChange={e =>
                setTrainingParams({ ...trainingParams, dropout: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2 рекомендуется: 0.4 (было: 0.3 в v1)
            </p>
          </div>

          {/* Label Smoothing */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Label Smoothing
              <Tooltip content="Смягчение меток. Предотвращает излишнюю уверенность модели. Рекомендуется: 0.1. Диапазон: 0-0.3.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="0.5"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.label_smoothing}
              onChange={e =>
                setTrainingParams({ ...trainingParams, label_smoothing: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2: новый параметр, рекомендуется 0.1
            </p>
          </div>

          {/* Focal Loss Gamma */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Focal Loss Gamma
              <Tooltip content="Параметр фокусировки на сложных примерах. Рекомендуется: 2.5. Больше = больше фокус на hard examples.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="5"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.focal_gamma}
              onChange={e =>
                setTrainingParams({ ...trainingParams, focal_gamma: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2 рекомендуется: 2.5 (было: 2.0 в v1)
            </p>
          </div>

          {/* Gaussian Noise Std */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Gaussian Noise Std
              <Tooltip content="Стандартное отклонение гауссовского шума для аугментации. Рекомендуется: 0.01. Добавляет робастность к шуму.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.001"
              min="0"
              max="0.1"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.gaussian_noise_std}
              onChange={e =>
                setTrainingParams({ ...trainingParams, gaussian_noise_std: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2: новый параметр, рекомендуется 0.01
            </p>
          </div>

          {/* Oversample Ratio */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Oversample Ratio
              <Tooltip content="Коэффициент оверсэмплинга для редких классов. Рекомендуется: 0.5. Помогает при дисбалансе классов.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.oversample_ratio}
              onChange={e =>
                setTrainingParams({ ...trainingParams, oversample_ratio: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2: новый параметр, рекомендуется 0.5
            </p>
          </div>

          {/* LR Scheduler T_0 */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Scheduler T_0 (Period)
              <Tooltip content="Период первого цикла для CosineAnnealing scheduler. Рекомендуется: 10. Определяет частоту перезапуска.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              min="1"
              max="100"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.scheduler_T_0}
              onChange={e =>
                setTrainingParams({ ...trainingParams, scheduler_T_0: parseInt(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2: новый параметр, рекомендуется 10
            </p>
          </div>

          {/* LR Scheduler T_mult */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Scheduler T_mult (Multiplier)
              <Tooltip content="Множитель периода для CosineAnnealing. Рекомендуется: 2. Увеличивает период на каждом цикле.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              min="1"
              max="10"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.scheduler_T_mult}
              onChange={e =>
                setTrainingParams({ ...trainingParams, scheduler_T_mult: parseInt(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2: новый параметр, рекомендуется 2
            </p>
          </div>
```

### 4. Обновить существующие поля с Tooltip

**Заменить существующие поля на версии с русскими подсказками:**

#### Epochs (строка ~822-840):
```tsx
          {/* Epochs */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Epochs
              <Tooltip content="Количество эпох обучения. Одна эпоха = один проход по всем данным. Рекомендуется: 150. Больше = дольше обучение.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              min="1"
              max="500"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.epochs}
              onChange={e =>
                setTrainingParams({ ...trainingParams, epochs: parseInt(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2 рекомендуется: 150 (было: 50 в v1)
            </p>
          </div>
```

#### Batch Size (строка ~842-858):
```tsx
          {/* Batch Size */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Batch Size
              <Tooltip content="Размер пакета данных. Влияет на стабильность градиентов и скорость. Рекомендуется: 256. Больше = стабильнее, но требует больше памяти.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              min="8"
              max="512"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.batch_size}
              onChange={e =>
                setTrainingParams({ ...trainingParams, batch_size: parseInt(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2 рекомендуется: 256 (было: 64 в v1)
            </p>
          </div>
```

#### Learning Rate (строка ~860-877):
```tsx
          {/* Learning Rate */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">
              Learning Rate
              <Tooltip content="Скорость обучения. Определяет размер шага оптимизации. Рекомендуется: 0.00005 (5e-5). Для финансовых данных нужен маленький LR.">
                <Info className="inline-block ml-1 h-3 w-3 text-gray-500 cursor-help" />
              </Tooltip>
            </label>
            <input
              type="number"
              step="0.00001"
              min="0.00001"
              max="0.1"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary"
              value={trainingParams.learning_rate}
              onChange={e =>
                setTrainingParams({ ...trainingParams, learning_rate: parseFloat(e.target.value) })
              }
              disabled={trainingStatus.is_training}
            />
            <p className="text-xs text-gray-500 mt-1">
              v2 рекомендуется: 0.00005 (было: 0.001 в v1) - КРИТИЧНО!
            </p>
          </div>
```

### 5. Добавить новые checkbox'ы в секцию Options (после строки ~910):

```tsx
        {/* Options */}
        <div className="space-y-3 mb-6">
          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.export_onnx}
              onChange={e =>
                setTrainingParams({ ...trainingParams, export_onnx: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
              Export to ONNX format (for optimized inference)
            </span>
          </label>

          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.auto_promote}
              onChange={e =>
                setTrainingParams({ ...trainingParams, auto_promote: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
              Auto-promote to production (if accuracy threshold is met)
            </span>
          </label>

          {/* НОВЫЕ CHECKBOX'Ы V2 */}
          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.use_augmentation}
              onChange={e =>
                setTrainingParams({ ...trainingParams, use_augmentation: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <Tooltip content="Включить аугментацию данных (Gaussian noise). Повышает робастность модели к шуму.">
              <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                Enable Data Augmentation (рекомендуется для v2)
              </span>
            </Tooltip>
          </label>

          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.use_focal_loss}
              onChange={e =>
                setTrainingParams({ ...trainingParams, use_focal_loss: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <Tooltip content="Использовать Focal Loss вместо CrossEntropy. Лучше справляется с дисбалансом классов.">
              <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                Use Focal Loss (рекомендуется для v2)
              </span>
            </Tooltip>
          </label>

          <label className="flex items-center gap-3 cursor-pointer group">
            <input
              type="checkbox"
              className="w-5 h-5 rounded border-gray-700 text-primary focus:ring-primary"
              checked={trainingParams.use_oversampling}
              onChange={e =>
                setTrainingParams({ ...trainingParams, use_oversampling: e.target.checked })
              }
              disabled={trainingStatus.is_training}
            />
            <Tooltip content="Увеличить количество примеров редких классов. Балансирует датасет.">
              <span className="text-sm text-gray-300 group-hover:text-white transition-colors cursor-help">
                Use Oversampling (рекомендуется для v2)
              </span>
            </Tooltip>
          </label>
        </div>
```

---

## Итого

**Добавлено новых полей:** 8 числовых input + 3 checkbox
**Обновлено существующих:** 3 поля с подсказками
**Все дефолтные значения:** Установлены на v2 оптимизированные

**Все параметры теперь:**
1. ✅ Имеют русские подсказки с объяснением
2. ✅ Показывают рекомендуемое значение
3. ✅ Имеют дефолтные v2 значения
4. ✅ Передаются в backend API

**Примечание:** Компонент Tooltip уже существует в файле (строки 136-156), поэтому просто используем его.
