"""Дымовые тесты: каждый скрипт должен импортироваться и рисовать кадры.

Анимации обычно не проверить в CI: FuncAnimation ничего не считает, пока нет
окна, а plt.show() под Agg просто возвращает управление. Поэтому здесь
FuncAnimation временно подменяется заглушкой, которая синхронно прогоняет
несколько кадров — так ловятся ошибки внутри update(), а не только импорта.

Запуск:  python -m pytest test_scripts.py -q
"""

import os
import runpy

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

import matplotlib.animation as animation
import matplotlib.pyplot as plt

#: Анимационные сцены: прогоняются кадры
ANIMATED = [
    "Photon.py",
    "Electron.py",
    "Proton.py",
    "DddBaryon.py",
    "Neutron.py",
    "SpinorSim.py",
]

#: Расчётные скрипты: запускаются целиком с ключами, подавляющими окна
COMPUTED = [
    ("alpha_running.py", ["--no-plot"]),
    ("bell.py", ["--no-plot", "--samples", "5000"]),
    ("photon_broadening.py", ["--no-plot"]),
    ("ProtonRadius.py", []),
    ("CubeSpace.py", []),
]

FRAMES_TO_CHECK = 4


@pytest.fixture
def headless(monkeypatch):
    """Подменяет FuncAnimation и plt.show, чтобы кадры считались синхронно."""
    rendered = []

    class FakeAnimation:
        def __init__(self, fig, func, frames=None, **kwargs):
            n = frames if isinstance(frames, int) else FRAMES_TO_CHECK
            for i in range(min(n, FRAMES_TO_CHECK)):
                func(i)
                rendered.append((func, i))

    monkeypatch.setattr(animation, "FuncAnimation", FakeAnimation)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(matplotlib, "use", lambda *a, **k: None)
    yield rendered
    plt.close("all")


@pytest.mark.parametrize("script", ANIMATED)
def test_animated_script_renders_frames(script, headless, monkeypatch):
    monkeypatch.setattr("sys.argv", [script])
    runpy.run_path(script, run_name="__main__")
    assert headless, f"{script}: FuncAnimation не был вызван — кадры не проверены"

    # ни один художник не должен содержать NaN в геометрии
    for fig_num in plt.get_fignums():
        for ax in plt.figure(fig_num).axes:
            for coll in ax.collections:
                seg = getattr(coll, "_segments3d", None)
                if seg is not None and len(np.asarray(seg)):
                    assert not np.isnan(np.asarray(seg, dtype=float)).any(), (
                        f"{script}: NaN в геометрии {type(coll).__name__}")


@pytest.mark.parametrize("script,argv", COMPUTED, ids=[s for s, _ in COMPUTED])
def test_computed_script_runs(script, argv, monkeypatch):
    monkeypatch.setattr("sys.argv", [script] + argv)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        plt.close("all")
