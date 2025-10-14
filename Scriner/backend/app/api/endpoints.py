# backend/app/api/endpoints.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from ..core.websocket_manager import manager
from ..core.logger import log
from ..core.bybit_client import BybitRESTClient

router = APIRouter()

@router.websocket("/ws/screener")
async def websocket_endpoint(websocket: WebSocket):
  """ Основная точка входа для WebSocket-соединений от фронтенда. """
  await manager.connect(websocket)
  try:
    while True:
      await websocket.receive_text()
  except WebSocketDisconnect:
    manager.disconnect(websocket)
    log.warning(f"WebSocketDisconnect: Соединение {websocket.client.host} закрыто.")
  except Exception as e:
    log.error(f"Произошла ошибка в WebSocket-соединении: {e}")
    manager.disconnect(websocket)

@router.get("/api/klines/{symbol}")
async def get_klines_for_symbol(symbol: str):
    """
    Предоставляет исторические данные свечей для указанного символа.
    """
    log.info(f"Запрос на получение klines для символа: {symbol}")
    client = BybitRESTClient()
    klines = await client.get_historical_klines(symbol)
    if klines is None:
        raise HTTPException(status_code=500, detail="Не удалось получить данные от Bybit API.")
    return klines
