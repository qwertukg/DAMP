"""Тесты DAMP: инварианты статьи и сверки быстрых путей с прямыми.

Системный слой лежит в system/, клиентский фреймворк damp_api — в client/. Тесты системы проверяют её изнутри,
тесты клиента — только через damp_api.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DAMP_LOG_FILE", "off")
# Тесты считают по-настоящему и не трогают data/cache; тесты кеша включают его сами во временной папке.
os.environ.setdefault("DAMP_CACHE", "off")

_ROOT = Path(__file__).resolve().parents[1]
for _layer in ("system", "client"):
    _path = str(_ROOT / _layer)
    if _path not in sys.path:
        sys.path.insert(0, _path)
