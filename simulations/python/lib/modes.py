"""Общие геометрические помощники для барионных/лептонных сцен.

Раньше эти три функции были скопированы в Proton.py, DddBaryon.py,
Neutron.py и SpinorSim.py — с расхождениями и опечатками между копиями.
"""

import numpy as np


def to_segments(x, y, z) -> np.ndarray:
    """Полилиния (N точек) -> массив отрезков (N-1, 2, 3) для Line3DCollection."""
    points = np.stack((x, y, z), axis=1).reshape(-1, 1, 3)
    return np.concatenate([points[:-1], points[1:]], axis=1)


def core_metric(x, y, R: float, r: float) -> np.ndarray:
    """Нормированная близость к горловине тора: 0 — горловина, 1 — внешний край.

    r_current пробегает от (R - r) в горловине до (R + r) на периферии,
    поэтому нормировка на 2r даёт ровно [0, 1].
    """
    r_current = np.hypot(x, y)
    return np.clip((r_current - (R - r)) / (2.0 * r), 0.0, 1.0)


def adaptive_indices(x, y, R: float, r: float,
                     base: int = 5, span: int = 25) -> np.ndarray:
    """Индексы точек с переменным шагом: густо в горловине, редко на периферии.

    Шаг = base + span * core_metric**2.
    """
    num_points = len(x)
    idx = []
    i = 0
    while i < num_points:
        idx.append(i)
        cm = core_metric(x[i], y[i], R, r)
        i += int(base + span * cm ** 2)
    return np.asarray(idx, dtype=int)


def alpha_profile(x, y, R: float, r: float,
                  peak: float = 0.85, drop: float = 0.60) -> np.ndarray:
    """Яркость стрелок: максимум в ядре, затухание к периферии."""
    return peak - drop * core_metric(x, y, R, r) ** 2
