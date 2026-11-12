# Инсайты по логам (DAMP)

Эталон: `layout.log`, `evaluate.log`, `log_samples/baseline_run2.log`, `log_samples/combined_run.log`, `log_samples/full_run.log`.  
Цель: `log_samples/baseline_run.log`.

Строк: **6879**, различных строк (стимулов): **233**, точек пространства: **1516** на сетке 42×42, детекторов: **856**, семейств событий: **131**, типов событий: **199**.

Карты: `log_insights.png` — псевдоцвета кодов (§5.14), `log_insights.energy.png` — с яркостью по энергии (рис. 18), `log_insights.types.png` — семейства событий.

Проверка эмбеддингов (§6.2): корреляция сходства эмбеддингов со сходством первичных кодов **0.565**; среднее сходство эмбеддингов у похожих строк 0.234, у остальных 0.002; строк без активных детекторов: 63.

## 1. Типы и семейства событий

### 1.1 Семейства (нижние слои иерархии детекторов: области и гиперкластеры, §7.1.10)

| # | семейство (общие слова) | видов строк | типов | строк |
|---|---|---|---|---|
| 0 | short * partitions (видов строк: 4) | 4 | 4 | 2223 |
| 1 | run (видов строк: 5) | 5 | 3 | 2134 |
| 3 | encoder (видов строк: 10) | 10 | 10 | 785 |
| 2 | energy (видов строк: 13) | 13 | 12 | 458 |
| 42 | continuity * swaps (видов строк: 2) | 2 | 2 | 199 |
| 43 | detectors init count * … | 1 | 1 | 140 |
| 4 | visual step * points … (видов строк: 3) | 3 | 2 | 130 |
| 5 | hierarchy (видов строк: 6) | 6 | 6 | 123 |
| 6 | adaptive radius step * … (видов строк: 2) | 2 | 2 | 75 |
| 7 | short pair sampling requested … | 1 | 1 | 40 |
| 24 | run (видов строк: 12) | 12 | 10 | 38 |
| 44 | sign | 1 | 1 | 35 |

### 1.2 Типы (верхние слои иерархии: кластеры и их части)

| # | тип (общие слова) | семейство | видов строк | строк | запусков | пример |
|---|---|---|---|---|---|---|
| 75 | short parallel * workers * partitions * chunk … | 0 | 1 | 2132 | 3 | layout.short.parallel pairs=144 workers=8 partitions=8 chunk_size=18 … |
| 0 | run progress step * mode accepted swaps evaluated … (видов строк: 2) | 1 | 2 | 1836 | 1 | layout.run.progress step=1 mode=long accepted_swaps=1106 evaluated_pa… |
| 1 | energy long progress step * prev radius (видов строк: 2) | 2 | 2 | 361 | 3 | layout.energy.long.progress step=1 energy=0.283335 energy_prev=None e… |
| 2 | run swap ratio step * swap ratio * … | 1 | 1 | 293 | 3 | layout.run.swap_ratio step=1 swap_ratio=0.430556 avg_ratio=0.430556 a… |
| 76 | encoder colour band dimension * layer index * … | 3 | 1 | 210 | 4 | encoder.colour_band dimension=Angle layer_index=0 colour_rank=0 band_… |
| 3 | continuity sweep sweep * swaps * gain * … | 42 | 1 | 196 | 1 | layout.continuity.sweep sweep=0 swaps=10 gain=2.38443 worst_defect=92… |
| 4 | detectors init count * overlap | 43 | 1 | 140 | 4 | detectors.init count=2 overlap=0.7 [4.1] |
| 77 | encoder layer dimension * dimension index * layer … | 3 | 1 | 140 | 4 | encoder.layer dimension=X dimension_index=1 layer_index=0 count=2 det… |
| 78 | visual step * points grid size energy diff … (видов строк: 2) | 4 | 2 | 107 | 2 | layout.visual step=151 points=1440 grid_size=38 energy=0.463622 energ… |
| 79 | encoder encode input stimuli * values * dimensions | 3 | 1 | 103 | 4 | encoder.encode.input stimuli=1 values=(0, 10, 10) dimensions=3 [4.6] |
| 80 | encoder encode normalized stimuli * values | 3 | 1 | 103 | 4 | encoder.encode.normalized stimuli=1 values=(0, 10, 10) [4.4] |
| 81 | encoder encode code stimuli * code length * … | 3 | 1 | 103 | 4 | encoder.encode.code stimuli=1 code_length=512 active_bits=90 active_b… |

### 1.3 Слои иерархии детекторов (§7.1.10, §8.2)

| λ_d | роль | детекторов | средний радиус | точек в поле | активных на строку |
|---|---|---|---|---|---|
| 0.5 | семейство | 51 | 1.69 | 9.9 | 1.2 |
| 0.6 | семейство | 64 | 1.55 | 8.5 | 1.2 |
| 0.7 | промежуточный | 113 | 1.25 | 5.9 | 1.5 |
| 0.75 | тип | 197 | 1.08 | 4.4 | 2.0 |
| 0.8 | тип | 139 | 1.04 | 4.2 | 1.3 |
| 0.85 | тип | 292 | 1.01 | 2.6 | 2.4 |

## 2. Новизна и аномалии

### 2.1 Новые строки целевых логов (§7.1.13.3)

Неизвестных: **58**, известных по частям: **6** (⚠ — в строке есть проблемное слово).

| статус | знакомость | детекторов | раз | где | строка | ближайшая известная |
|---|---|---|---|---|---|---|
| неизвестна ⚠ | 0.33 | 0 | 1 | log_samples/baseline_run.log:348 | Traceback (most recent call last): | — |
| по частям ⚠ | 0.41 | 2 | 1 | log_samples/baseline_run.log:414 | RuntimeError: | 2026-11-12T07:08:04.660986Z ERROR re_grpc_client::write: Write messages call fa… |
| по частям ⚠ | 0.48 | 2 | 1 | log_samples/baseline_run.log:411 | raise RuntimeError(''' | 2026-11-12T07:08:04.660986Z ERROR re_grpc_client::write: Write messages call fa… |
| по частям ⚠ | 0.53 | 2 | 1 | log_samples/baseline_run.log:432 | 2026-11-12T07:12:05.754669Z ERROR re_grpc_server: Error while receiving messages: h2 protocol error: error re… | 2026-11-12T07:08:04.660986Z ERROR re_grpc_client::write: Write messages call fa… |
| неизвестна | 0.26 | 0 | 2 | log_samples/baseline_run.log:374 | ...<19 lines>... | — |
| неизвестна | 0.28 | 0 | 1 | log_samples/baseline_run.log:375 | **_sparse_layout_options(), | — |
| неизвестна | 0.28 | 0 | 1 | log_samples/baseline_run.log:386 | timeout=timeout, | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:372 | return Layout( | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:404 | return Popen(process_obj) | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:408 | _check_not_importing_main() | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:422 | if __name__ == '__main__': | — |
| неизвестна | 0.29 | 0 | 1 | log_samples/baseline_run.log:423 | freeze_support() | — |

_…и ещё 52; полный список — в JSON._

### 2.2 Новые значения чисел (§4.2.1, §2.2.2)

_нет_

### 2.3 Строки эталона вне структуры: энергия всех их точек ниже порога (§5.7, §5.14)

| энергия | раз | запусков | где | строка |
|---|---|---|---|---|
| 0.0 | 2 | 2 | log_samples/combined_run.log:153 | "singularities": 2, |
| 0.025 | 2 | 2 | log_samples/combined_run.log:154 | "neighbour_pairs": 2206, |
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

- `log_samples/full_run.log:459-498` — сходство 0.78
- `log_samples/combined_run.log:295-334` — сходство 0.63
- `log_samples/baseline_run2.log:141-180` — сходство 0.59

### Сходство запусков

|  | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0: layout.log | 1.0 | 0.14 | 0.11 | 0.08 | 0.12 | 0.01 |
| 1: evaluate.log | 0.14 | 1.0 | 0.04 | 0.04 | 0.04 | 0.0 |
| 2: log_samples/baseline_run2.log | 0.11 | 0.04 | 1.0 | 0.82 | 0.93 | 0.38 |
| 3: log_samples/combined_run.log | 0.08 | 0.04 | 0.82 | 1.0 | 0.78 | 0.47 |
| 4: log_samples/full_run.log | 0.12 | 0.04 | 0.93 | 0.78 | 1.0 | 0.37 |
| 5: log_samples/baseline_run.log | 0.01 | 0.0 | 0.38 | 0.47 | 0.37 | 1.0 |

## 4. Связи событий (§2.3.1, §2.3.2)

### 4.1 «После A обычно идёт B»

| после | идёт | поддержка | уверенность | подъём |
|---|---|---|---|---|
| run progress step * mode accepted swaps evaluated … (видов строк: 2) | energy long progress step * prev radius (видов строк: 2) | 300 | 0.99 | 7.4 |
| energy long progress step * prev radius (видов строк: 2) | run progress step * mode accepted swaps evaluated … (видов строк: 2) | 361 | 0.82 | 7.4 |
| encoder layer dimension * dimension index * layer … | encoder colour band dimension * layer index * … | 140 | 1.0 | 12.9 |
| encoder colour band dimension * layer index * … | encoder layer dimension * dimension index * layer … | 210 | 0.6 | 11.7 |
| encoder encode input stimuli * values * dimensions | encoder encode normalized stimuli * values | 103 | 1.0 | 26.4 |
| encoder encode normalized stimuli * values | encoder encode code stimuli * code length * … | 103 | 1.0 | 26.4 |
| encoder encode code stimuli * code length * … | encoder encode input stimuli * values * dimensions | 103 | 0.93 | 24.6 |
| encoder layer dimension angle dimension index * layer … | encoder colour band dimension * layer index * … | 70 | 1.0 | 12.9 |
| adaptive radius step * pair radius * swap … | run swap ratio step * swap ratio * … | 72 | 0.82 | 12.1 |
| short partition requested * produced * partitions * … | short parallel * workers * partitions * chunk … | 44 | 1.0 | 17.2 |
| short tiles partitions * row parts * col … | short partition requested * produced * partitions * … | 44 | 0.89 | 54.8 |
| short pair sampling requested * generated * rejected … | short parallel * workers * partitions * chunk … | 40 | 0.88 | 15.1 |

### 4.2 Что предшествует повторяющимся проблемным событиям (до 3 событий назад)

| событие | раз | предвестники (подъём, раз) |
|---|---|---|
| projects damp venv lib site packages rerun dynamic … (видов строк: 2) | 10 | similarity threshold lambda threshold * eta * eta … (×271.3, 5); energy long kernel kernel distance radius min * … (×271.3, 5); energy distance long distance euclidean short distance squared … (×271.3, 5) |
| gpu import failed error no module named moderngl | 5 | gpu pair energy unavailable error no module named … (×542.6, 5); distance eps distance eps * вне статьи (×542.6, 5); codes code length * ones min * ones … (×542.6, 5) |
| gpu tensor unavailable error no module named torch | 5 | gpu pair energy unavailable error no module named … (×542.6, 5); distance eps distance eps * вне статьи (×542.6, 5); gpu import failed error no module named moderngl (×542.6, 5) |
| gpu pair energy unavailable error no module named … | 5 | distance eps distance eps * вне статьи (×542.6, 5); gpu config use gpu true (×542.6, 5); codes code length * ones min * ones … (×542.6, 5) |

