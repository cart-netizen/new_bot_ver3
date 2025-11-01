"""
Гибридная CNN-LSTM модель для предсказания движений рынка.

Архитектура:
- CNN блоки для извлечения локальных паттернов из временных рядов
- BiLSTM для моделирования долгосрочных зависимостей
- Attention механизм для фокусировки на важных временных шагах
- Multi-task learning: direction + confidence + expected return

Путь: backend/ml_engine/models/hybrid_cnn_lstm.py
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
  """Конфигурация модели."""
  input_features: int = 110  # OrderBook(50) + Candle(25) + Indicator(35)
  sequence_length: int = 60  # 60 временных шагов (~30 секунд при 0.5s)

  # CNN параметры
  cnn_channels: Tuple[int, ...] = (64, 128, 256)
  cnn_kernel_sizes: Tuple[int, ...] = (3, 5, 7)

  # LSTM параметры
  lstm_hidden: int = 256
  lstm_layers: int = 2
  lstm_dropout: float = 0.2

  # Attention параметры
  attention_units: int = 128

  # Output параметры
  num_classes: int = 3  # BUY=1, HOLD=0, SELL=2

  # Regularization
  dropout: float = 0.3


class CNNBlock(nn.Module):
  """CNN блок для извлечения локальных паттернов."""

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      dropout: float = 0.3
  ):
    super().__init__()

    self.conv = nn.Conv1d(
      in_channels=in_channels,
      out_channels=out_channels,
      kernel_size=kernel_size,
      padding=kernel_size // 2  # same padding
    )
    self.batch_norm = nn.BatchNorm1d(out_channels)
    self.activation = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    logger.debug(
      f"Инициализирован CNN блок: in={in_channels}, out={out_channels}, "
      f"kernel={kernel_size}"
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass.

    Args:
        x: (batch, channels, sequence)

    Returns:
        Tensor (batch, out_channels, sequence//2)
    """
    x = self.conv(x)
    x = self.batch_norm(x)
    x = self.activation(x)
    x = self.dropout(x)
    x = self.pool(x)
    return x


class AttentionLayer(nn.Module):
  """Attention механизм для фокусировки на важных временных шагах."""

  def __init__(self, hidden_size: int, attention_units: int):
    super().__init__()

    self.attention = nn.Sequential(
      nn.Linear(hidden_size, attention_units),
      nn.Tanh(),
      nn.Linear(attention_units, 1)
    )

    logger.debug(
      f"Инициализирован Attention: hidden={hidden_size}, "
      f"units={attention_units}"
    )

  def forward(
      self,
      lstm_output: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass.

    Args:
        lstm_output: (batch, sequence, hidden_size)

    Returns:
        context: (batch, hidden_size)
        attention_weights: (batch, sequence)
    """
    # Вычисляем attention scores
    attention_scores = self.attention(lstm_output)  # (batch, seq, 1)
    attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq, 1)

    # Weighted sum
    context = torch.sum(
      attention_weights * lstm_output,
      dim=1
    )  # (batch, hidden_size)

    return context, attention_weights.squeeze(-1)


class HybridCNNLSTM(nn.Module):
  """
  Гибридная CNN-LSTM модель с multi-task learning.

  Архитектура:
  1. CNN блоки извлекают локальные паттерны
  2. BiLSTM моделирует временные зависимости
  3. Attention фокусируется на важных моментах
  4. Три выходных головы:
     - Direction classifier (BUY/HOLD/SELL)
     - Confidence regressor (0-1)
     - Expected return regressor (%)
  """

  def __init__(self, config: ModelConfig):
    super().__init__()

    self.config = config

    # ==================== CNN БЛОКИ ====================
    cnn_blocks = []
    in_channels = 1  # Начинаем с 1 канала (unsqueezed features)

    for i, (out_channels, kernel_size) in enumerate(
        zip(config.cnn_channels, config.cnn_kernel_sizes)
    ):
      cnn_blocks.append(
        CNNBlock(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=kernel_size,
          dropout=config.dropout
        )
      )
      in_channels = out_channels

    self.cnn_blocks = nn.ModuleList(cnn_blocks)

    # Размер после CNN (sequence уменьшается в 2^n раз из-за pooling)
    cnn_output_size = config.cnn_channels[-1]
    reduced_sequence = config.sequence_length // (2 ** len(config.cnn_channels))

    # ==================== LSTM ====================
    self.lstm = nn.LSTM(
      input_size=cnn_output_size,
      hidden_size=config.lstm_hidden,
      num_layers=config.lstm_layers,
      batch_first=True,
      dropout=config.lstm_dropout if config.lstm_layers > 1 else 0,
      bidirectional=True
    )

    lstm_output_size = config.lstm_hidden * 2  # Bidirectional

    # ==================== ATTENTION ====================
    self.attention = AttentionLayer(
      hidden_size=lstm_output_size,
      attention_units=config.attention_units
    )

    # ==================== OUTPUT HEADS ====================

    # Direction classifier (BUY/HOLD/SELL)
    self.direction_head = nn.Sequential(
      nn.Linear(lstm_output_size, 128),
      nn.ReLU(),
      nn.Dropout(config.dropout),
      nn.Linear(128, config.num_classes)
    )

    # Confidence regressor (0-1)
    self.confidence_head = nn.Sequential(
      nn.Linear(lstm_output_size, 64),
      nn.ReLU(),
      nn.Dropout(config.dropout),
      nn.Linear(64, 1),
      nn.Sigmoid()  # Ограничиваем [0, 1]
    )

    # Expected return regressor
    self.return_head = nn.Sequential(
      nn.Linear(lstm_output_size, 64),
      nn.ReLU(),
      nn.Dropout(config.dropout),
      nn.Linear(64, 1)
    )

    # Инициализация весов
    self._initialize_weights()

    logger.info(
      f"Инициализирована HybridCNNLSTM модель: "
      f"input_features={config.input_features}, "
      f"sequence_length={config.sequence_length}, "
      f"cnn_channels={config.cnn_channels}, "
      f"lstm_hidden={config.lstm_hidden}, "
      f"lstm_layers={config.lstm_layers}"
    )

  def _initialize_weights(self):
    """Инициализация весов модели."""
    for module in self.modules():
      if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(
          module.weight,
          mode='fan_out',
          nonlinearity='relu'
        )
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)
      elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)
      elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

  def forward(
      self,
      x: torch.Tensor,
      return_attention: bool = False
  ) -> Dict[str, torch.Tensor]:
    """
    Forward pass.

    Args:
        x: (batch, sequence_length, input_features)
        return_attention: Возвращать ли attention weights

    Returns:
        Dict с выходами:
            - direction_logits: (batch, num_classes)
            - confidence: (batch, 1)
            - expected_return: (batch, 1)
            - attention_weights: (batch, sequence) если return_attention=True
    """
    batch_size = x.size(0)

    # ==================== CNN PROCESSING ====================
    # Добавляем channel dimension для Conv1d
    # (batch, sequence, features) -> (batch, 1, sequence * features)
    x = x.reshape(batch_size, 1, -1)

    # Пропускаем через CNN блоки
    for cnn_block in self.cnn_blocks:
      x = cnn_block(x)

    # ==================== LSTM PROCESSING ====================
    # Переформатируем для LSTM
    # (batch, channels, reduced_sequence) -> (batch, reduced_sequence, channels)
    x = x.transpose(1, 2)

    # BiLSTM
    lstm_out, _ = self.lstm(x)  # (batch, reduced_sequence, lstm_hidden*2)

    # ==================== ATTENTION ====================
    context, attention_weights = self.attention(lstm_out)

    # ==================== OUTPUT HEADS ====================
    direction_logits = self.direction_head(context)
    confidence = self.confidence_head(context)
    expected_return = self.return_head(context)

    outputs = {
      'direction_logits': direction_logits,
      'confidence': confidence,
      'expected_return': expected_return
    }

    if return_attention:
      outputs['attention_weights'] = attention_weights

    return outputs

  def predict(
      self,
      x: torch.Tensor,
      temperature: float = 1.0
  ) -> Dict[str, torch.Tensor]:
    """
    Inference с temperature scaling для калибровки вероятностей.

    Args:
        x: (batch, sequence_length, input_features)
        temperature: Temperature для softmax (default=1.0)

    Returns:
        Dict с предсказаниями:
            - direction: (batch,) класс предсказания
            - direction_probs: (batch, num_classes) вероятности
            - confidence: (batch,) уверенность
            - expected_return: (batch,) ожидаемая доходность
    """
    self.eval()

    with torch.no_grad():
      outputs = self.forward(x, return_attention=False)

      # Direction с temperature scaling
      direction_logits = outputs['direction_logits'] / temperature
      direction_probs = torch.softmax(direction_logits, dim=-1)
      direction = torch.argmax(direction_probs, dim=-1)

      return {
        'direction': direction,
        'direction_probs': direction_probs,
        'confidence': outputs['confidence'].squeeze(-1),
        'expected_return': outputs['expected_return'].squeeze(-1)
      }

  def get_model_size(self) -> Dict[str, int]:
    """Получить размер модели."""
    total_params = sum(p.numel() for p in self.parameters())
    trainable_params = sum(
      p.numel() for p in self.parameters() if p.requires_grad
    )

    return {
      'total_params': total_params,
      'trainable_params': trainable_params,
      'non_trainable_params': total_params - trainable_params
    }


def create_model(config: Optional[ModelConfig] = None) -> HybridCNNLSTM:
  """
  Фабрика для создания модели.

  Args:
      config: Конфигурация модели (None = default)

  Returns:
      Инициализированная модель
  """
  if config is None:
    config = ModelConfig()

  model = HybridCNNLSTM(config)

  model_size = model.get_model_size()
  logger.info(
    f"Создана модель с {model_size['total_params']:,} параметрами "
    f"({model_size['trainable_params']:,} обучаемых)"
  )

  return model


# Пример использования
if __name__ == "__main__":
  # Создаем модель с дефолтной конфигурацией
  model = create_model()

  # Тестовые данные
  batch_size = 32
  sequence_length = 60
  input_features = 110

  x = torch.randn(batch_size, sequence_length, input_features)

  # Forward pass
  outputs = model(x, return_attention=True)

  print("Outputs:")
  for key, value in outputs.items():
    print(f"  {key}: {value.shape}")

  # Inference
  predictions = model.predict(x)

  print("\nPredictions:")
  for key, value in predictions.items():
    print(f"  {key}: {value.shape}")