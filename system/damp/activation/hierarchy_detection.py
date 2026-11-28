"""Пакетное детектирование иерархией самоорганизующихся детекторов (§6.1, §6.5).

ActivationEngine считает реакцию кодового пространства на один набор стимулов и пишет её в журнал — так
читается первая раскладка картинкой. Уровню иерархии нужно то же для сотен тысяч кодов по одному: какие
детекторы зажигает каждый код. Здесь те же формулы считаются пакетами на устройстве:

    a_j = sim_λa(s, v_j)                         — пороговая активация точки стимулом (§6.1);
    Ê_a(j) = Σ a_v / d(j, v) по R(j, r_e) / max  — энергия активации точки: формула §5.7 на матрице активации
                                                   (расширение статьи); точка с Ê_a < μ_a гаснет — одиночные
                                                   зажжённые точки на отшибе молчат;
    E(d) = Σ a_j · e_j / e_d по точкам поля d    — уровень активации детектора, точки с e_j ≥ μ_e (§6.5);
    детектор активен при E(d) ≥ μ_d.

На одном стимуле результат совпадает с ActivationEngine.detect.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from damp.activation.activation import Detector
from damp.article_refs import STIMULUS_DETECTION
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER


class HierarchyDetection:
    """Какие детекторы иерархии зажигает каждый код-стимул (§6.5), пакетами.

    Детекторы упорядочены по слоям: слой — порог λ_d, с которым детектор построен (§6.4), слой 0 — самый
    низкий порог (самые широкие области, §7.1.10). Номера детекторов в ответе — по этому порядку.
    """

    BLOCK = 2048
    # Строк в одном умножении «активация × поля детекторов» и «активация × соседи». Видеокарта выбирает ядро
    # умножения по числу строк, и порядок сложения в float32 менялся бы от размера пачки. Поэтому пачка идёт частями
    # ровно по ROWS строк (последняя дополняется нулями): ядро всегда одно, и результат строки зависит только от неё
    # самой — не от того, сколько строк идёт рядом. Число небольшое: один рисунок дашборда — около сотни окон, и
    # дополнять их до тысяч строк значило бы считать в десятки раз больше нужного.
    ROWS = 256
    # Ячеек матрицы «вхождения стимулов × точки карты» за один проход по наборам (float32: 4 байта на ячейку).
    SET_CELLS = 48_000_000

    def __init__(
        self,
        layout: Layout,
        detectors: Sequence[Detector],
        energies: Sequence[float],
        *,
        similarity: str,
        lambda_threshold: float,
        eta: float | None,
        energy_threshold: float,
        detector_threshold: float,
        activate_best: bool = False,
        local_energy_threshold: float = 0.0,
        local_energy_radius: float = 0.0,
        use_gpu: bool = True,
        int8_codes: bool = False,
        name: str = "layout",
    ) -> None:
        import torch

        if not detectors:
            raise ValueError("at least one detector is required")
        if similarity not in ("cosine", "jaccard"):
            raise ValueError("similarity must be cosine or jaccard")
        table = layout.code_table
        self._length = int(table.code_length)
        self._row_bytes = -(-self._length // 8)
        if len(energies) != int(table.packed.shape[0]):
            raise ValueError("energies size must match layout points")
        thresholds = sorted({float(detector.lambda_threshold) for detector in detectors})
        layer = np.asarray([thresholds.index(float(detector.lambda_threshold)) for detector in detectors], dtype=np.int64)
        order = np.argsort(layer, kind="stable")
        self.detectors = tuple(detectors[int(index)] for index in order)
        self.layer_index = layer[order]
        self.layer_thresholds = tuple(thresholds)
        self._torch = torch
        self._device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self._similarity = similarity
        self._lambda = float(lambda_threshold)
        self._eta = None if eta is None else float(eta)
        self._detector_threshold = float(detector_threshold)
        # Самая похожая точка зажигается всегда, даже если её сходство ниже λ_a: код без единой зажжённой точки
        # не оставил бы следа на карте (расширение статьи).
        self._activate_best = bool(activate_best)
        self._name = str(name)
        # Коды на устройстве (вне статьи, техника счёта): int8 и целочисленное произведение int8 × int8 → int32
        # (torch._int_mm) дают то же число общих бит (§2.2.4), что произведение float32, — срабатывания те же до бита;
        # памяти под коды точек вчетверо меньше (карта в 28 тыс. точек: 0.29 ГБ вместо 1.14 ГБ), произведение быстрее.
        # Первые точки сверяются с подсчётом бит по упакованным кодам; не сошлось или не CUDA — float32, как раньше.
        packed_points = np.ascontiguousarray(table.packed)
        self._point_count = int(packed_points.shape[0])
        self._int8 = bool(int8_codes) and self._device.type == "cuda" and hasattr(torch, "_int_mm")
        self._points = self._to_device(packed_points, points=True)
        if self._int8 and not self._int8_matches(packed_points):
            self._int8 = False
            self._points = self._to_device(packed_points, points=True)
        if self._int8:
            self._ones = self._points[: self._point_count].sum(dim=1, dtype=torch.int32).to(torch.float32)
        else:
            self._ones = self._points.sum(dim=1)
        energy = np.asarray(energies, dtype=np.float32)
        # Точки с энергией ниже μ_e в уровень активации детектора не входят (§6.5).
        weights = np.where(energy >= float(energy_threshold), energy, 0.0).astype(np.float32)
        self._weights = torch.as_tensor(weights, device=self._device)
        cells = layout.cells_array().astype(np.float64)
        centre = np.asarray([detector.center for detector in self.detectors], dtype=np.float64)
        radius = np.asarray([detector.radius for detector in self.detectors], dtype=np.float64)
        # Поле детектора — точки не дальше r_d от его центра (§6.5), как в ActivationEngine.
        coverage, covered = self._coverage_matrix(cells, centre, radius)
        # Поля — нули и единицы: на хосте они uint8 (у карты в 29 тыс. точек с 12 тыс. детекторов — 0.35 ГБ, а float32 вместе с
        # транспонированием держал 2.8 ГБ ОЗУ — пик прогона, 2026-11-24), во float32 — уже на устройстве; значения те же.
        self._coverage = torch.as_tensor(coverage, device=self._device).to(torch.float32)
        # Энергия активации точки (расширение статьи): формула энергии точки §5.7, где вместо сходства с соседями —
        # их активация a_v, с нормировкой на наибольшую по карте для того же стимула (Ê = E / E_max, §5.7). Точка с
        # Ê_a ниже μ_a гаснет: одиночная зажжённая точка, вокруг которой ничего не горит, в детекторы не идёт. В §6.5
        # энергия e_j — неподвижная энергия раскладки и одиночных точек не гасит. 0 — выключено (как в статье).
        self._local_threshold = float(local_energy_threshold)
        self._neighbours = None
        if self._local_threshold > 0.0:
            if float(local_energy_radius) <= 0.0:
                raise ValueError("local_energy_radius must be positive when local_energy_threshold is set")
            from scipy.spatial import cKDTree

            pairs = cKDTree(cells).query_pairs(r=float(local_energy_radius), output_type="ndarray")
            first = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int64)
            second = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int64)
            weight = (1.0 / np.hypot(cells[first, 0] - cells[second, 0], cells[first, 1] - cells[second, 1])).astype(np.float32)
            self._neighbours = torch.sparse_coo_tensor(
                torch.as_tensor(np.stack([first, second])), torch.as_tensor(weight), (int(cells.shape[0]), int(cells.shape[0]))
            ).coalesce().to(self._device)
        self._local_radius = float(local_energy_radius)
        self._detector_energy = torch.as_tensor(
            np.asarray([detector.energy for detector in self.detectors], dtype=np.float32), device=self._device
        )
        LOGGER.event(
            "activation.hierarchy.init",
            section=STIMULUS_DETECTION,
            data={
                "name": self._name,
                "points": int(cells.shape[0]),
                "detectors": len(self.detectors),
                "layers": len(thresholds),
                "layer_thresholds": list(thresholds),
                "detectors_per_layer": np.bincount(self.layer_index, minlength=len(thresholds)).tolist(),
                "covered_points_mean": float(covered.mean()),
                "lambda_threshold": self._lambda,
                "eta": 0.0 if self._eta is None else self._eta,
                "energy_threshold": float(energy_threshold),
                "detector_threshold": self._detector_threshold,
                "activate_best": self._activate_best,
                "local_energy_threshold": self._local_threshold,
                "local_energy_radius": self._local_radius,
                "local_energy_pairs": 0 if self._neighbours is None else int(self._neighbours.values().shape[0]),
                "codes": "int8" if self._int8 else "float32",
                "note": "детекторы упорядочены по слоям: слой 0 — самый низкий порог λ_d (самые широкие области); "
                "точка с энергией активации Ê_a < μ_a гаснет (§5.7 на матрице активации, расширение статьи; 0 — выключено)",
            },
        )

    @staticmethod
    def _coverage_matrix(cells: "np.ndarray", centre: "np.ndarray", radius: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        """Поля детекторов: матрица «точки × детекторы» uint8 из 0/1 и число точек в поле каждого детектора.

        Тот же признак hypot(cell − centre) ≤ r (float64, те же биты на границе), но без матрицы N × D float64:
        для каждого детектора hypot считается только по точкам прямоугольника радиуса (hypot ≥ |dx|, поэтому вне
        прямоугольника признак ложен; прямоугольник взят с запасом, лишние кандидаты отсеивает сам hypot)."""
        count, detectors = int(cells.shape[0]), int(centre.shape[0])
        order = np.argsort(cells[:, 0], kind="stable")
        xs, ys = cells[order, 0], cells[order, 1]
        transposed = np.zeros((detectors, count), dtype=np.uint8)
        covered = np.zeros(detectors, dtype=np.int64)
        for number in range(detectors):
            cx, cy, r = float(centre[number, 0]), float(centre[number, 1]), float(radius[number])
            margin = 1e-9 * (1.0 + abs(cx) + abs(r))
            low = int(np.searchsorted(xs, cx - r - margin, side="left"))
            high = int(np.searchsorted(xs, cx + r + margin, side="right"))
            if high <= low:
                continue
            inside = np.hypot(xs[low:high] - cx, ys[low:high] - cy) <= r
            transposed[number, order[low:high][inside]] = 1
            covered[number] = int(inside.sum())
        return np.ascontiguousarray(transposed.T), covered

    # torch._int_mm: левая матрица — больше 16 строк, длина кода и число точек — кратны 8 (дополняются нулями: нулевой
    # бит и нулевая точка общих бит не добавляют).
    INT_MM_ROWS_MIN = 17
    INT_MM_MULTIPLE = 8
    INT8_CHECK_ROWS = 8

    def _to_device(self, packed: "np.ndarray", points: bool = False):  # noqa: ANN202
        """Коды на устройстве: float32 (строки × длина) либо, при int8, int8 с длиной, дополненной до кратной 8;
        points — это коды точек карты: у них и число строк дополняется до кратного 8."""
        torch = self._torch
        bits = np.unpackbits(packed, axis=1, count=self._length)
        if not self._int8:
            return torch.as_tensor(bits, device=self._device).to(torch.float32)
        rows, length = int(bits.shape[0]), int(bits.shape[1])
        multiple = self.INT_MM_MULTIPLE
        codes = torch.zeros(((-(-rows // multiple) * multiple) if points else rows, -(-length // multiple) * multiple),
                            dtype=torch.int8, device=self._device)
        codes[:rows, :length] = torch.as_tensor(bits, device=self._device).to(torch.int8)
        return codes

    def _common(self, block):  # noqa: ANN001, ANN202
        """Число общих бит (§2.2.4) кодов блока с кодами точек карты, float32 (строки × точки): целые числа, одни и те
        же у произведения float32 и у целочисленного int8 × int8 → int32."""
        torch = self._torch
        if block.dtype != torch.int8:
            return block @ self._points.T
        taken = int(block.shape[0])
        if taken == 0:
            return torch.zeros((0, self._point_count), dtype=torch.float32, device=self._device)
        if taken < self.INT_MM_ROWS_MIN:
            block = torch.cat((block, block.new_zeros((self.INT_MM_ROWS_MIN - taken, block.shape[1]))))
        return torch._int_mm(block, self._points.T)[:taken, : self._point_count].to(torch.float32)

    def _int8_matches(self, packed_points: "np.ndarray") -> bool:
        """Сверка int8-произведения с подсчётом бит по упакованным кодам: несколько точек карты со всеми точками."""
        torch = self._torch
        count = self._point_count
        if count == 0 or not hasattr(np, "bitwise_count"):
            return count == 0
        probe = np.unique(np.linspace(0, count - 1, min(count, self.INT8_CHECK_ROWS)).astype(np.int64))
        try:
            with torch.no_grad():
                found = self._common(self._points[torch.as_tensor(probe, device=self._device)]).cpu().numpy()
        except Exception:  # noqa: BLE001 — нет целочисленного произведения на этом устройстве
            return False
        expected = np.stack([np.bitwise_count(packed_points & packed_points[row]).sum(axis=1) for row in probe])
        return bool(np.array_equal(found.astype(np.int64), expected.astype(np.int64)))

    def active(self, packed: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        """Пары (номер кода, номер детектора) для всех сработавших детекторов; коды — строки × ⌈L/8⌉ байт."""
        torch = self._torch
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if packed.ndim != 2 or packed.shape[1] != self._row_bytes:
            raise ValueError("codes must be rows × ceil(code_length / 8) packed bytes")
        rows, columns, lit, dropped = [], [], 0, 0
        with torch.no_grad():
            for start in range(0, packed.shape[0], self.BLOCK):
                raw = self._activation(self._to_device(packed[start : start + self.BLOCK]))
                activation = self._local(raw)
                lit += int((activation > 0).sum().item())
                dropped += int((raw > 0).sum().item()) - int((activation > 0).sum().item())
                level = self._level(activation)
                found = torch.nonzero(level >= self._detector_threshold)
                rows.append(found[:, 0].cpu().numpy() + start)
                columns.append(found[:, 1].cpu().numpy())
        rows = np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, dtype=np.int64)
        columns = np.concatenate(columns).astype(np.int64) if columns else np.zeros(0, dtype=np.int64)
        per_code = np.bincount(rows, minlength=packed.shape[0])
        LOGGER.event(
            "activation.hierarchy.detect",
            section=STIMULUS_DETECTION,
            data={
                "name": self._name,
                "codes": int(packed.shape[0]),
                "points_lit_per_code": lit / max(1, packed.shape[0]),
                "points_dropped_by_local_energy_per_code": dropped / max(1, packed.shape[0]),
                "detectors_per_code": float(per_code.mean()) if per_code.size else 0.0,
                "detectors_per_code_by_layer": [float(count) / max(1, packed.shape[0]) for count in self._per_layer(columns)],
                "codes_without_detectors": float(np.mean(per_code == 0)) if per_code.size else 0.0,
                "note": "код зажигает все точки со сходством не ниже λ_a (§6.1); детектор активен, если энергия "
                "зажжённых точек его поля не меньше доли μ_d его собственной энергии (§6.5)",
            },
        )
        return rows, columns

    def lit(self, packed: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Зажжённые точки карты (§6.1): тройки (номер кода, номер точки, активация a_j > 0) — та же активация, что
        в active и active_sets (после энергии активации, если она включена); без свёртки в уровни детекторов. Для
        показа карты отклика (вне статьи)."""
        torch = self._torch
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if packed.ndim != 2 or packed.shape[1] != self._row_bytes:
            raise ValueError("codes must be rows × ceil(code_length / 8) packed bytes")
        rows, points, values = [], [], []
        with torch.no_grad():
            for start in range(0, packed.shape[0], self.BLOCK):
                activation = self._local(self._activation(self._to_device(packed[start : start + self.BLOCK])))
                found = torch.nonzero(activation > 0)
                rows.append(found[:, 0].cpu().numpy() + start)
                points.append(found[:, 1].cpu().numpy())
                values.append(activation[found[:, 0], found[:, 1]].cpu().numpy())
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        return (np.concatenate(rows).astype(np.int64), np.concatenate(points).astype(np.int64),
                np.concatenate(values).astype(np.float32))

    def active_with_lit(
        self, packed: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        """Срабатывания и зажжённые точки за один проход: (код, детектор, уровень активации) и (код, точка, a_j).

        То же, что active_sets для наборов из одного кода (набор i — код i) и lit для тех же кодов, цифра в цифру,
        но активация карты (§6.1) и энергия активации считаются один раз, а не дважды. Нужно пути одной картинки
        (дашборд): там каждое окно — свой набор, и карта отклика показывается вместе со срабатываниями.
        """
        torch = self._torch
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if packed.ndim != 2 or packed.shape[1] != self._row_bytes:
            raise ValueError("codes must be rows × ceil(code_length / 8) packed bytes")
        rows, columns, levels, lit_rows, lit_points, lit_values = [], [], [], [], [], []
        with torch.no_grad():
            for start in range(0, packed.shape[0], self.BLOCK):
                activation = self._local(self._activation(self._to_device(packed[start : start + self.BLOCK])))
                lit = torch.nonzero(activation > 0)
                lit_rows.append(lit[:, 0].cpu().numpy() + start)
                lit_points.append(lit[:, 1].cpu().numpy())
                lit_values.append(activation[lit[:, 0], lit[:, 1]].cpu().numpy())
                level = self._level(activation)
                found = torch.nonzero(level >= self._detector_threshold)
                rows.append(found[:, 0].cpu().numpy() + start)
                columns.append(found[:, 1].cpu().numpy())
                levels.append(level[found[:, 0], found[:, 1]].cpu().numpy())

        def joined(parts: list, dtype) -> "np.ndarray":  # noqa: ANN001
            return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

        return (joined(rows, np.int64), joined(columns, np.int64), joined(levels, np.float32),
                joined(lit_rows, np.int64), joined(lit_points, np.int64), joined(lit_values, np.float32))

    def _host(self, tensor) -> "np.ndarray":  # noqa: ANN001
        """Тензор устройства — в новый массив, чью память держит numpy, а не распределитель torch; числа те же, что у
        .cpu().numpy(). Мелкие куски из .cpu() распределитель torch на Windows после освобождения системе не отдаёт: на
        146 млн троек уровня 3 (50 000 картинок, ~1330 проходов по наборам) процесс держал лишние ~1.7 ГБ до конца уровня
        (замер 2026-11-26 на тех же кусках: после склейки .cpu() — 4.91 ГиБ, копия в numpy — 3.26 ГиБ)."""
        torch = self._torch
        kind = {torch.int32: np.int32, torch.int64: np.int64, torch.float32: np.float32}[tensor.dtype]
        found = np.empty(tuple(tensor.shape), dtype=kind)
        if found.size:
            torch.from_numpy(found).copy_(tensor)
        return found

    LAYER_CHUNK = 1 << 24

    def _per_layer(self, columns: "np.ndarray") -> "np.ndarray":
        """Сработавших детекторов каждого слоя (для журнала) — кусками столбца троек: без массива слоёв на все тройки
        разом (на 146 млн троек он брал +1.2 ГБ ОЗУ на пике, замер 2026-11-26). Те же числа, что у count_nonzero."""
        layers = len(self.layer_thresholds)
        found = np.zeros(layers, dtype=np.int64)
        for start in range(0, int(np.shape(columns)[0]), self.LAYER_CHUNK):
            found += np.bincount(self.layer_index[columns[start : start + self.LAYER_CHUNK]], minlength=layers)[:layers]
        return found

    @staticmethod
    def _joined(parts: list, dtype) -> "np.ndarray":  # noqa: ANN001
        """Куски одного столбца троек — сразу в один массив dtype (без промежуточной склейки и копии astype), куски
        отдаются тут же. На 2.5 млн окон уровня 0 (40 000 картинок) прежняя склейка держала разом куски, их склейку и
        копию — +3.2 ГБ ОЗУ к проходу (замер 2026-11-26)."""
        if not parts:
            return np.zeros(0, dtype=dtype)
        found = np.concatenate(parts, dtype=dtype)
        parts.clear()
        return found

    def _by_rows(self, activation, compute):  # noqa: ANN001, ANN202
        """compute над строками activation частями ровно по ROWS строк (последняя часть дополняется нулевыми
        строками): ядро умножения одно и то же, результат строки зависит только от неё самой (вне статьи — техника
        счёта; проверочные картинки на обучающие окна не влияют бит в бит)."""
        torch = self._torch
        count = int(activation.shape[0])
        parts = []
        for start in range(0, max(count, 1), self.ROWS):
            part = activation[start : start + self.ROWS]
            rows = int(part.shape[0])
            if rows < self.ROWS:
                part = torch.cat([part, part.new_zeros((self.ROWS - rows, part.shape[1]))])
            parts.append(compute(part)[:rows])
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def _level(self, activation):  # noqa: ANN001, ANN202
        """Уровень активации детекторов E(d) = Σ a_j · e_j / e_d по полю детектора (§6.5), строка за строкой
        (частями по ROWS строк, см. _by_rows)."""
        return self._by_rows(
            activation, lambda part: ((part * self._weights[None, :]) @ self._coverage) / self._detector_energy[None, :]
        )

    def _local(self, activation):  # noqa: ANN001, ANN202
        """Энергия активации (расширение статьи): Ê_a(j) = E_a(j) / max E_a, E_a(j) = Σ a_v / d(j, v) по точкам
        R(j, r_e) без самой j — формула §5.7, где вместо сходства с соседями их активация. Точки с Ê_a < μ_a гаснут.

        Счёт — частями по ROWS строк, как в _level: порядок сложения не зависит от размера пачки.
        """
        if self._neighbours is None:
            return activation
        torch = self._torch
        energy = self._by_rows(activation, lambda part: torch.sparse.mm(self._neighbours, part.T).T)
        top = energy.max(dim=1, keepdim=True).values
        relative = torch.where(top > 0, energy / torch.clamp(top, min=1e-12), torch.zeros_like(energy))
        return torch.where(relative >= self._local_threshold, activation, torch.zeros_like(activation))

    def _activation(self, block):  # noqa: ANN001, ANN202
        """Пороговая активация точек карты кодами блока (§6.1): строки — коды, столбцы — точки."""
        torch = self._torch
        ones = block.sum(dim=1, dtype=torch.int32).to(torch.float32) if block.dtype == torch.int8 else block.sum(dim=1)
        common = self._common(block)
        if self._similarity == "cosine":
            denominator = torch.sqrt(ones[:, None] * self._ones[None, :])
        else:
            denominator = ones[:, None] + self._ones[None, :] - common
        similarity = torch.where(denominator > 0, common / denominator, torch.zeros_like(common))
        if self._eta is None:
            activation = torch.where(similarity >= self._lambda, similarity, torch.zeros_like(similarity))
        else:
            activation = similarity / (1.0 + torch.exp(-self._eta * (similarity - self._lambda)))
        if self._activate_best:
            best = similarity.max(dim=1, keepdim=True).values
            activation = torch.maximum(activation, torch.where(similarity >= best, similarity, torch.zeros_like(similarity)))
        return activation

    def active_sets(
        self,
        packed: "np.ndarray",
        index: "np.ndarray",
        groups: "np.ndarray",
        group_count: int,
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Детекторы, сработавшие на наборы стимулов (§6.5): тройки (набор, детектор, уровень активации).

        packed — различные коды стимулов; index — номер кода у каждого вхождения стимула в набор; groups —
        номер набора у каждого вхождения, по возрастанию. Реакции карты на стимулы набора объединяются:
        a_j = max по s ∈ S от sim_λa(s, v_j); дальше, как для одного стимула, E(d) = Σ a_j · e_j / e_d и
        детектор активен при E(d) ≥ μ_d. Набор из одного стимула даёт то же, что active.

        packed может лежать на диске (damp.rows_on_disk.RowsOnDisk — коды окон уровня после отбора образцов карты):
        тогда коды читаются только кусками прохода, packed[номера] — те же байты, что у массива в ОЗУ.
        """
        torch = self._torch
        if not getattr(packed, "on_disk", False):
            packed = np.ascontiguousarray(packed, dtype=np.uint8)
        elif packed.dtype != np.uint8:
            raise ValueError("codes must be rows × ceil(code_length / 8) packed bytes")
        index = np.asarray(index, dtype=np.int64)
        groups = np.asarray(groups, dtype=np.int64)
        if packed.ndim != 2 or packed.shape[1] != self._row_bytes:
            raise ValueError("codes must be rows × ceil(code_length / 8) packed bytes")
        if index.shape != groups.shape or index.ndim != 1:
            raise ValueError("index and groups must be one-dimensional arrays of the same length")
        if groups.size and (np.any(np.diff(groups) < 0) or groups[0] < 0 or groups[-1] >= group_count):
            raise ValueError("groups must be ascending set numbers below group_count")
        starts = np.searchsorted(groups, np.arange(int(group_count) + 1))
        # Вхождений в одном проходе — столько, чтобы матрица «вхождения × точки» помещалась в память устройства.
        limit = max(64, self.SET_CELLS // max(1, self._point_count))
        # Куски номеров — int32 (пока номера наборов и детекторов меньше 2³¹), итог — int64, те же числа (_joined).
        narrow = torch.int32 if max(int(group_count), int(self._coverage.shape[1])) < 2**31 else torch.int64
        rows, columns, levels, lit, dropped = [], [], [], 0, 0
        first = 0
        with torch.no_grad():
            while first < group_count:
                last = int(np.searchsorted(starts, starts[first] + limit, side="right")) - 1
                last = min(int(group_count), max(last, first + 1))
                entries = slice(int(starts[first]), int(starts[last]))
                if entries.stop > entries.start:
                    codes, inverse = np.unique(index[entries], return_inverse=True)
                    inverse = inverse.reshape(-1)
                    activation = self._activation(self._to_device(packed[codes]))
                    members = groups[entries]
                    # Порядковый номер вхождения внутри своего набора: вхождения с одним номером лежат в разных
                    # наборах, и максимум по точкам берётся за столько шагов, сколько стимулов в самом большом наборе.
                    rank = np.arange(entries.start, entries.stop) - starts[members]
                    # Реакции на стимулы набора объединяются максимумом по точкам (§6.5).
                    combined = torch.zeros((last - first, activation.shape[1]), dtype=activation.dtype, device=self._device)
                    for step in range(int(rank.max()) + 1):
                        chosen = np.flatnonzero(rank == step)
                        local = torch.as_tensor(members[chosen] - first, device=self._device)
                        picked = activation[torch.as_tensor(inverse[chosen], device=self._device)]
                        combined[local] = torch.maximum(combined[local], picked)
                    # Энергия активации — по объединённой реакции набора (расширение статьи; 0 — выключено).
                    before = int((combined > 0).sum().item())
                    combined = self._local(combined)
                    lit += int((combined > 0).sum().item())
                    dropped += before - int((combined > 0).sum().item())
                    level = self._level(combined)
                    found = torch.nonzero(level >= self._detector_threshold)
                    rows.append(self._host((found[:, 0] + first).to(narrow)))
                    columns.append(self._host(found[:, 1].to(narrow)))
                    levels.append(self._host(level[found[:, 0], found[:, 1]]))
                first = last
        rows = self._joined(rows, np.int64)
        columns = self._joined(columns, np.int64)
        levels = self._joined(levels, np.float32)
        per_set = np.bincount(rows, minlength=int(group_count))
        LOGGER.event(
            "activation.hierarchy.detect_sets",
            section=STIMULUS_DETECTION,
            data={
                "name": self._name,
                "sets": int(group_count),
                "stimuli": int(index.shape[0]),
                "stimuli_per_set": float(index.shape[0]) / max(1, int(group_count)),
                "distinct_codes": int(packed.shape[0]),
                "points_lit_per_set": lit / max(1, int(group_count)),
                "points_dropped_by_local_energy_per_set": dropped / max(1, int(group_count)),
                "detectors_per_set": float(per_set.mean()) if per_set.size else 0.0,
                "detectors_per_set_by_layer": [float(count) / max(1, int(group_count)) for count in self._per_layer(columns)],
                "sets_without_detectors": float(np.mean(per_set == 0)) if per_set.size else 0.0,
                "note": "реакции карты на стимулы набора объединяются максимумом по точкам; детектор активен, "
                "если энергия зажжённых точек его поля не меньше доли μ_d его собственной энергии (§6.5)",
            },
        )
        return rows, columns, levels


