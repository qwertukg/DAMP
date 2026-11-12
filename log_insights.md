# Инсайты по логам (DAMP)

Эталон: `layout.log`, `evaluate.log`, `log_samples/baseline_run2.log`, `log_samples/combined_run.log`, `log_samples/full_run.log`.  
Цель: `log_samples/baseline_run.log`.

Строк: **6879**, различных строк (стимулов): **243**, точек пространства: **1625** на сетке 44×44, детекторов: **880**, семейств событий: **112**, типов событий: **193**.

Карты: `log_insights.png` — псевдоцвета кодов (§5.14), `log_insights.energy.png` — с яркостью по энергии (рис. 18), `log_insights.types.png` — семейства событий.

Проверка эмбеддингов (§6.2): корреляция сходства эмбеддингов со сходством первичных кодов **0.821**; среднее сходство эмбеддингов у похожих строк 0.37, у остальных 0.036; строк без активных детекторов: 61.

## 1. Типы и семейства событий

### 1.1 Семейства (нижние слои иерархии детекторов: области и гиперкластеры, §7.1.10)

| # | семейство (общие слова) | видов строк | типов | строк |
|---|---|---|---|---|
| 0 | layout (видов строк: 85) | 85 | 55 | 5470 |
| 2 | encoder (видов строк: 7) | 7 | 5 | 633 |
| 1 | detectors (видов строк: 4) | 4 | 4 | 167 |
| 4 | hierarchy (видов строк: 5) | 5 | 4 | 122 |
| 3 | encoder encode code stimuli … | 1 | 1 | 103 |
| 5 | sign | 1 | 1 | 35 |
| 15 | layout (видов строк: 10) | 10 | 9 | 25 |
| 10 | decoder (видов строк: 6) | 6 | 4 | 24 |
| 7 | dimension init title min … (видов строк: 2) | 2 | 1 | 21 |
| 8 | encoder random bits dimension … (видов строк: 2) | 2 | 2 | 21 |
| 9 | encoder long range order … (видов строк: 2) | 2 | 1 | 21 |
| 6 | activation detector layer lambda … | 1 | 1 | 18 |

### 1.2 Типы (верхние слои иерархии: кластеры и их части)

| # | тип (общие слова) | семейство | видов строк | строк | запусков | пример |
|---|---|---|---|---|---|---|
| 0 | layout (видов строк: 4) | 0 | 4 | 2182 | 4 | layout.short.parallel pairs=144 workers=8 partitions=8 chunk_size=18 … |
| 1 | layout run progress step mode accepted swaps evaluated … (видов строк: 2) | 0 | 2 | 1836 | 1 | layout.run.progress step=1 mode=long accepted_swaps=1106 evaluated_pa… |
| 2 | layout energy long progress step prev radius (видов строк: 2) | 0 | 2 | 356 | 3 | layout.energy.long.progress step=26 energy=0.360289 energy_prev=0.283… |
| 3 | layout run swap ratio step swap ratio avg … | 0 | 1 | 293 | 3 | layout.run.swap_ratio step=1 swap_ratio=0.430556 avg_ratio=0.430556 a… |
| 7 | encoder colour band dimension layer index rank start … (видов строк: 2) | 2 | 2 | 210 | 4 | encoder.colour_band dimension=X layer_index=0 colour_rank=0 band_star… |
| 8 | encoder encode stimuli values (видов строк: 2) | 2 | 2 | 206 | 4 | encoder.encode.input stimuli=1 values=(0, 10, 10) dimensions=3 [4.6] |
| 4 | layout continuity sweep sweep swaps gain worst defect | 0 | 1 | 196 | 1 | layout.continuity.sweep sweep=0 swaps=10 gain=2.38443 worst_defect=92… |
| 5 | detectors init count overlap | 1 | 1 | 140 | 4 | detectors.init count=2 overlap=0.7 [4.1] |
| 6 | encoder layer dimension dimension index layer index count … | 2 | 1 | 140 | 4 | encoder.layer dimension=X dimension_index=1 layer_index=0 count=2 det… |
| 9 | encoder encode code stimuli code length active bits … | 3 | 1 | 103 | 4 | encoder.encode.code stimuli=1 code_length=512 active_bits=90 active_b… |
| 10 | hierarchy composition encode features active bits code length … | 4 | 1 | 100 | 1 | hierarchy.composition.encode features=9 active_bits=875 code_length=2… |
| 11 | layout adaptive radius step pair swap rate stage … (видов строк: 2) | 0 | 2 | 75 | 3 | layout.adaptive.radius step=201 pair_radius=19 swap_rate=9.3 stage_in… |

### 1.3 Слои иерархии детекторов (§7.1.10, §8.2)

| λ_d | роль | детекторов | средний радиус | точек в поле | активных на строку |
|---|---|---|---|---|---|
| 0.5 | семейство | 44 | 1.73 | 10.8 | 5.4 |
| 0.6 | семейство | 63 | 1.58 | 9.4 | 4.6 |
| 0.7 | промежуточный | 116 | 1.25 | 6.0 | 3.9 |
| 0.75 | тип | 187 | 1.1 | 4.6 | 3.0 |
| 0.8 | тип | 157 | 1.06 | 4.3 | 1.8 |
| 0.85 | тип | 313 | 1.01 | 2.6 | 2.6 |

## 2. Новизна и аномалии

### 2.1 Новые строки целевых логов (§7.1.13.3)

Неизвестных: **57**, известных по частям: **7** (⚠ — в строке есть проблемное слово).

| статус | знакомость | детекторов | раз | где | строка | ближайшая известная |
|---|---|---|---|---|---|---|
| неизвестна ⚠ | 0.3 | 0 | 1 | log_samples/baseline_run.log:348 | Traceback (most recent call last): | — |
| по частям ⚠ | 0.41 | 1 | 1 | log_samples/baseline_run.log:414 | RuntimeError: | 2026-11-12T07:08:04.660986Z ERROR re_grpc_client::write: Write messages call fa… |
| по частям ⚠ | 0.47 | 1 | 1 | log_samples/baseline_run.log:411 | raise RuntimeError(''' | 2026-11-12T07:08:04.660986Z ERROR re_grpc_client::write: Write messages call fa… |
| по частям ⚠ | 0.53 | 1 | 1 | log_samples/baseline_run.log:432 | 2026-11-12T07:12:05.754669Z ERROR re_grpc_server: Error while receiving messages: h2 protocol error: error re… | 2026-11-12T07:08:04.660986Z ERROR re_grpc_client::write: Write messages call fa… |
| неизвестна | 0.26 | 0 | 2 | log_samples/baseline_run.log:374 | ...<19 lines>... | — |
| неизвестна | 0.28 | 0 | 1 | log_samples/baseline_run.log:422 | if __name__ == '__main__': | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:350 | from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=3032, pipe_handle=932) | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:386 | timeout=timeout, | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:408 | _check_not_importing_main() | — |
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

- `log_samples/full_run.log:459-498` — сходство 0.81
- `log_samples/baseline_run2.log:141-180` — сходство 0.75
- `log_samples/combined_run.log:315-354` — сходство 0.74

### Сходство запусков

|  | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0: layout.log | 1.0 | 0.22 | 0.14 | 0.12 | 0.16 | 0.13 |
| 1: evaluate.log | 0.22 | 1.0 | 0.02 | 0.02 | 0.02 | 0.04 |
| 2: log_samples/baseline_run2.log | 0.14 | 0.02 | 1.0 | 0.8 | 0.94 | 0.55 |
| 3: log_samples/combined_run.log | 0.12 | 0.02 | 0.8 | 1.0 | 0.81 | 0.59 |
| 4: log_samples/full_run.log | 0.16 | 0.02 | 0.94 | 0.81 | 1.0 | 0.54 |
| 5: log_samples/baseline_run.log | 0.13 | 0.04 | 0.55 | 0.59 | 0.54 | 1.0 |

## 4. Связи событий (§2.3.1, §2.3.2)

### 4.1 «После A обычно идёт B»

| после | идёт | поддержка | уверенность | подъём |
|---|---|---|---|---|
| layout run progress step mode accepted swaps evaluated … (видов строк: 2) | layout energy long progress step prev radius (видов строк: 2) | 300 | 0.98 | 6.9 |
| layout energy long progress step prev radius (видов строк: 2) | layout run progress step mode accepted swaps evaluated … (видов строк: 2) | 356 | 0.82 | 6.9 |
| encoder layer dimension dimension index layer index count … | encoder colour band dimension layer index rank start … (видов строк: 2) | 140 | 1.0 | 12.0 |
| encoder colour band dimension layer index rank start … (видов строк: 2) | encoder layer dimension dimension index layer index count … | 210 | 0.6 | 10.8 |
| encoder encode stimuli values (видов строк: 2) | encoder encode code stimuli code length active bits … | 103 | 1.0 | 24.4 |
| encoder encode code stimuli code length active bits … | encoder encode stimuli values (видов строк: 2) | 103 | 0.93 | 22.7 |
| encoder layer dimension angle dimension index layer index … | encoder colour band dimension layer index rank start … (видов строк: 2) | 70 | 1.0 | 12.0 |
| layout adaptive radius step pair swap rate stage … (видов строк: 2) | layout run swap ratio step swap ratio avg … | 75 | 0.81 | 11.1 |
| layout visual step points grid size energy diff … (видов строк: 2) | layout run swap ratio step swap ratio avg … | 63 | 0.81 | 11.0 |
| layout short pair sampling pairs requested pairs generated … | layout (видов строк: 4) | 40 | 1.0 | 15.3 |
| layout short tiles partitions row parts col parts … | layout (видов строк: 4) | 44 | 0.89 | 13.6 |
| layout visual step points grid size energy none … | layout (видов строк: 4) | 23 | 0.74 | 11.3 |

### 4.2 Что предшествует повторяющимся проблемным событиям (до 3 событий назад)

| событие | раз | предвестники (подъём, раз) |
|---|---|---|
| projects damp venv lib site packages rerun dynamic … (видов строк: 2) | 10 | layout similarity threshold lambda threshold eta eta mode … (×250.4, 5); layout energy long kernel kernel distance radius min … (×250.4, 5); layout energy distance long distance euclidean short distance … (×250.4, 5) |
| layout gpu error no module named (видов строк: 3) | 5 | layout distance eps distance eps вне статьи (×500.8, 5); layout codes code length ones min ones max … (×500.8, 5); layout gpu (видов строк: 2) (×250.4, 5) |

