# DAMP MNIST

Текущий режим эксперимента использует 1000 обучающих и 1000 тестовых
изображений MNIST.

## Построение Layout и базы эмбеддингов

Linux/macOS:

```bash
python3 main.py layout --count 1000
```

Windows PowerShell с Rerun:

```powershell
$env:RERUN='on'
python main.py layout --count 1000
```

Параметр `--count 1000` задаёт число обучающих изображений для построения
Layout. Если `--activation-samples` не указан, число обучающих эмбеддингов
автоматически равно `--count`, то есть также составляет 1000.

Прогресс этапов и частичные эмбеддинги атомарно сохраняются рядом с раскладкой
в `*.checkpoint.json`. Повторный запуск продолжает активацию с последнего
сохранённого батча. Размер батча, число потоков и интервал checkpoint задаются
константами `ACTIVATION_BATCH_SIZE`, `ACTIVATION_PARALLEL_WORKERS` и
`ACTIVATION_CHECKPOINT_EVERY` в `main.py`.

В Rerun полная последовательность процесса доступна на timeline `event`.
Итоговая раскладка остаётся видимой в `layout/final/image` и
`layout/final/points` независимо от выбранного момента времени.

## Оценка на 1000 тестовых изображениях

Linux/macOS:

```bash
python3 main.py evaluate --count 1000 --split test --limit 1000
```

Windows PowerShell:

```powershell
$env:RERUN='on'
python main.py evaluate --count 1000 --split test --limit 1000
```

Таким образом, полный эксперимент выполняется в режиме `1000 train / 1000 test`.
