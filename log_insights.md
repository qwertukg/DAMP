# Инсайты по логам (DAMP)

Эталон: `layout.log`, `evaluate.log`, `log_samples/baseline_run2.log`, `log_samples/combined_run.log`, `log_samples/full_run.log`.  
Цель: `log_samples/baseline_run.log`.

Строк: **6879**, различных строк (стимулов): **234**, точек пространства: **1541** на сетке 43×43, детекторов: **790**, семейств событий: **119**, типов событий: **219**.

Карты: `log_insights.png` — псевдоцвета кодов (§5.14), `log_insights.energy.png` — с яркостью по энергии (рис. 18), `log_insights.types.png` — семейства событий.

Проверка эмбеддингов (§6.2): корреляция сходства эмбеддингов со сходством первичных кодов **0.773**; среднее сходство эмбеддингов у похожих строк 0.334, у остальных 0.028; строк без активных детекторов: 61.

## 1. Типы и семейства событий

### 1.1 Семейства (нижние слои иерархии детекторов: области и гиперкластеры, §7.1.10)

| # | семейство (общие слова) | видов строк | типов | строк |
|---|---|---|---|---|
| 0 | layout (видов строк: 77) | 77 | 70 | 5448 |
| 1 | encoder (видов строк: 8) | 8 | 8 | 624 |
| 2 | detectors count * overlap (видов строк: 2) | 2 | 2 | 154 |
| 3 | encoder layer dimension * … | 1 | 1 | 140 |
| 4 | hierarchy * (видов строк: 4) | 4 | 4 | 121 |
| 5 | sign | 1 | 1 | 35 |
| 6 | layout * (видов строк: 4) | 4 | 4 | 32 |
| 10 | decoder * (видов строк: 6) | 6 | 5 | 24 |
| 8 | dimension init title * … (видов строк: 2) | 2 | 2 | 21 |
| 9 | encoder long range order … (видов строк: 2) | 2 | 2 | 21 |
| 7 | activation detector layer lambda … | 1 | 1 | 18 |
| 19 | pinwheel * (видов строк: 3) | 3 | 3 | 13 |

### 1.2 Типы (верхние слои иерархии: кластеры и их части)

| # | тип (общие слова) | семейство | видов строк | строк | запусков | пример |
|---|---|---|---|---|---|---|
| 0 | layout short parallel pairs * workers * partitions … | 0 | 1 | 2132 | 3 | layout.short.parallel pairs=144 workers=8 partitions=8 chunk_size=18 … |
| 1 | layout run progress step * mode accepted swaps … (видов строк: 2) | 0 | 2 | 1836 | 1 | layout.run.progress step=1 mode=long accepted_swaps=1106 evaluated_pa… |
| 2 | layout energy long progress step * prev radius (видов строк: 2) | 0 | 2 | 361 | 3 | layout.energy.long.progress step=1 energy=0.283335 energy_prev=None e… |
| 3 | layout run swap ratio step * swap ratio … | 0 | 1 | 293 | 3 | layout.run.swap_ratio step=1 swap_ratio=0.430556 avg_ratio=0.430556 a… |
| 4 | encoder colour band dimension * layer index * … | 1 | 1 | 210 | 4 | encoder.colour_band dimension=Angle layer_index=0 colour_rank=0 band_… |
| 5 | layout continuity sweep sweep * swaps * gain … | 0 | 1 | 196 | 1 | layout.continuity.sweep sweep=0 swaps=10 gain=2.38443 worst_defect=92… |
| 6 | detectors init count * overlap | 2 | 1 | 140 | 4 | detectors.init count=2 overlap=0.7 [4.1] |
| 7 | encoder layer dimension * dimension index * layer … | 3 | 1 | 140 | 4 | encoder.layer dimension=X dimension_index=1 layer_index=0 count=2 det… |
| 8 | layout visual step * points grid size energy … (видов строк: 2) | 0 | 2 | 107 | 2 | layout.visual step=151 points=1440 grid_size=38 energy=0.463622 energ… |
| 9 | encoder encode input stimuli * values * dimensions | 1 | 1 | 103 | 4 | encoder.encode.input stimuli=1 values=(0, 10, 10) dimensions=3 [4.6] |
| 10 | encoder encode normalized stimuli * values | 1 | 1 | 103 | 4 | encoder.encode.normalized stimuli=1 values=(0, 10, 10) [4.4] |
| 11 | encoder encode code stimuli * code length * … | 1 | 1 | 103 | 4 | encoder.encode.code stimuli=1 code_length=512 active_bits=90 active_b… |

### 1.3 Слои иерархии детекторов (§7.1.10, §8.2)

| λ_d | роль | детекторов | средний радиус | точек в поле | активных на строку |
|---|---|---|---|---|---|
| 0.5 | семейство | 42 | 1.63 | 9.3 | 4.1 |
| 0.6 | семейство | 64 | 1.58 | 8.8 | 5.7 |
| 0.7 | промежуточный | 105 | 1.31 | 6.4 | 3.5 |
| 0.75 | тип | 165 | 1.1 | 4.6 | 2.8 |
| 0.8 | тип | 128 | 1.06 | 4.3 | 1.6 |
| 0.85 | тип | 286 | 1.01 | 2.6 | 2.6 |

## 2. Новизна и аномалии

### 2.1 Новые строки целевых логов (§7.1.13.3)

Неизвестных: **57**, известных по частям: **7** (⚠ — в строке есть проблемное слово).

| статус | знакомость | детекторов | раз | где | строка | ближайшая известная |
|---|---|---|---|---|---|---|
| неизвестна ⚠ | 0.3 | 0 | 1 | log_samples/baseline_run.log:348 | Traceback (most recent call last): | — |
| по частям ⚠ | 0.4 | 1 | 1 | log_samples/baseline_run.log:414 | RuntimeError: | 2026-11-12T07:08:04.660986Z ERROR re_grpc_client::write: Write messages call fa… |
| по частям ⚠ | 0.47 | 1 | 1 | log_samples/baseline_run.log:411 | raise RuntimeError(''' | 2026-11-12T07:08:04.660986Z ERROR re_grpc_client::write: Write messages call fa… |
| по частям ⚠ | 0.53 | 1 | 1 | log_samples/baseline_run.log:432 | 2026-11-12T07:12:05.754669Z ERROR re_grpc_server: Error while receiving messages: h2 protocol error: error re… | 2026-11-12T07:08:04.660986Z ERROR re_grpc_client::write: Write messages call fa… |
| неизвестна | 0.26 | 0 | 2 | log_samples/baseline_run.log:374 | ...<19 lines>... | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:350 | from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=3032, pipe_handle=932) | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:386 | timeout=timeout, | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:408 | _check_not_importing_main() | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:422 | if __name__ == '__main__': | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:429 | To fix this issue, refer to the "Safe importing of main module" | — |
| неизвестна | 0.3 | 0 | 1 | log_samples/baseline_run.log:349 | File "<string>", line 1, in <module> | — |
| неизвестна | 0.3 | 0 | 1 | log_samples/baseline_run.log:385 | itertools.batched(zip(*iterables), chunksize), | — |

_…и ещё 52; полный список — в JSON._

### 2.2 Новые значения чисел (§4.2.1, §2.2.2)

_нет_

### 2.3 Строки эталона вне структуры: энергия всех их точек ниже порога (§5.7, §5.14)

| энергия | раз | запусков | где | строка |
|---|---|---|---|---|
| 0.0 | 2 | 2 | log_samples/combined_run.log:153 | "singularities": 2, |
| 0.0 | 35 | 2 | log_samples/combined_run.log:159 | "sign": -1 |

## 3. Инциденты и похожие окна эталона (§6.5, §2.2.5)

### log_samples/baseline_run.log:321-432

Строк: 92, из них сигналов: 65 (64 видов); первый сигнал — `log_samples/baseline_run.log:348`.

Сигналы:

- неизвестна ⚠ ×1: `Traceback (most recent call last):`
- по частям ⚠ ×1: `raise RuntimeError('''`
- по частям ⚠ ×1: `RuntimeError:`
- по частям ⚠ ×1: `2026-11-12T07:12:05.754669Z ERROR re_grpc_server: Error while receiving messages: h2 prot…`

Перед первым сигналом (ближайшее событие — первым):

- `layout.sim_cache.gpu use_gpu=true [5.12]`
- `layout.sim_cache.parallel workers=14 [5.10]`
- `layout.sim_cache.similarity similarity=cosine [2.2.4]`

Похожие окна эталона:

- `log_samples/full_run.log:459-498` — сходство 0.82
- `log_samples/combined_run.log:315-354` — сходство 0.74
- `log_samples/baseline_run2.log:141-180` — сходство 0.74

### Сходство запусков

|  | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0: layout.log | 1.0 | 0.18 | 0.15 | 0.15 | 0.15 | 0.13 |
| 1: evaluate.log | 0.18 | 1.0 | 0.02 | 0.02 | 0.02 | 0.04 |
| 2: log_samples/baseline_run2.log | 0.15 | 0.02 | 1.0 | 0.82 | 0.93 | 0.48 |
| 3: log_samples/combined_run.log | 0.15 | 0.02 | 0.82 | 1.0 | 0.8 | 0.55 |
| 4: log_samples/full_run.log | 0.15 | 0.02 | 0.93 | 0.8 | 1.0 | 0.47 |
| 5: log_samples/baseline_run.log | 0.13 | 0.04 | 0.48 | 0.55 | 0.47 | 1.0 |

## 4. Связи событий (§2.3.1, §2.3.2)

### 4.1 «После A обычно идёт B»

| после | идёт | поддержка | уверенность | подъём |
|---|---|---|---|---|
| layout run progress step * mode accepted swaps … (видов строк: 2) | layout energy long progress step * prev radius (видов строк: 2) | 300 | 0.99 | 7.5 |
| layout energy long progress step * prev radius (видов строк: 2) | layout run progress step * mode accepted swaps … (видов строк: 2) | 361 | 0.82 | 7.5 |
| encoder layer dimension * dimension index * layer … | encoder colour band dimension * layer index * … | 140 | 1.0 | 13.0 |
| encoder colour band dimension * layer index * … | encoder layer dimension * dimension index * layer … | 210 | 0.6 | 11.7 |
| encoder encode input stimuli * values * dimensions | encoder encode normalized stimuli * values | 103 | 1.0 | 26.5 |
| encoder encode normalized stimuli * values | encoder encode code stimuli * code length * … | 103 | 1.0 | 26.5 |
| encoder encode code stimuli * code length * … | encoder encode input stimuli * values * dimensions | 103 | 0.93 | 24.7 |
| encoder layer dimension angle dimension index * layer … | encoder colour band dimension * layer index * … | 70 | 1.0 | 13.0 |
| layout adaptive radius step * pair radius * … | layout run swap ratio step * swap ratio … | 72 | 0.82 | 12.2 |
| layout short partition pairs requested * produced * … | layout short parallel pairs * workers * partitions … | 44 | 1.0 | 17.3 |
| layout short tiles partitions * row parts * … | layout short partition pairs requested * produced * … | 44 | 0.89 | 55.0 |
| layout short pair sampling pairs requested * pairs … | layout short parallel pairs * workers * partitions … | 40 | 0.88 | 15.1 |

### 4.2 Что предшествует повторяющимся проблемным событиям (до 3 событий назад)

| событие | раз | предвестники (подъём, раз) |
|---|---|---|
| projects damp venv lib site packages rerun dynamic … (видов строк: 2) | 10 | layout similarity threshold lambda threshold * eta * … (×272.3, 5); layout energy long kernel kernel distance radius min … (×272.3, 5); layout energy distance long distance euclidean short distance … (×272.3, 5) |
| layout gpu pair energy unavailable error no module … | 5 | layout distance eps distance eps * вне статьи (×544.6, 5); layout codes code length * ones min * … (×544.6, 5); layout gpu config use gpu true (×544.6, 5) |
| layout gpu import failed error no module named … | 5 | layout gpu pair energy unavailable error no module … (×544.6, 5); layout distance eps distance eps * вне статьи (×544.6, 5); layout codes code length * ones min * … (×544.6, 5) |
| layout gpu tensor unavailable error no module named … | 5 | layout gpu import failed error no module named … (×544.6, 5); layout gpu pair energy unavailable error no module … (×544.6, 5); layout distance eps distance eps * вне статьи (×544.6, 5) |

