"""Регрессионные тесты на исправления из CODE_REVIEW.md.

Запуск:  python -m pytest test_simulations.py -q
(тесты не открывают окон — бэкенд принудительно Agg)
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

import lib.plot as plot
from lib.modes import adaptive_indices, alpha_profile, core_metric, to_segments
from lib.torus import Torus


@pytest.fixture(scope="module")
def ax():
    return plot.init("test", zoom=2.0, elev=17.0, azim=123.0, size=5.0)


# --- §1.2: init() обязан применять свои аргументы -------------------------

def test_init_applies_view_and_size(ax):
    assert (ax.elev, ax.azim) == (17.0, 123.0)
    assert plot.SIZE == 5.0              # раньше оставалось 8.0
    assert ax.get_xlim() == (-5.0, 5.0)


def test_init_theme_sets_colors():
    plot.init("light", theme='Light')
    assert (plot.FG_COLOR, plot.BG_COLOR) == ('black', 'white')
    plot.init("dark", theme='Dark')
    assert (plot.FG_COLOR, plot.BG_COLOR) == ('white', 'black')


# --- §1.3: Arrows не должен давать NaN ни для какого направления ----------

@pytest.mark.parametrize("v", [
    (0.0, 1.0, 0.0),    # строго вдоль +Y  — раньше NaN
    (0.0, -1.0, 0.0),   # строго вдоль -Y  — раньше NaN
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.3, -0.7, 0.2),
])
def test_arrows_no_nan(ax, v):
    a = plot.Arrows(ax, 1, plot.rgba('yellow'))
    z = np.zeros(1)
    a.recalc(z, z, z, np.array([v[0]]), np.array([v[1]]), np.array([v[2]]))
    assert not np.isnan(a._geom).any()


def test_arrows_head_is_symmetric(ax):
    """Четыре луча зонтика равноудалены от острия."""
    a = plot.Arrows(ax, 3, plot.rgba('yellow'))
    o = np.zeros(3)
    vx = np.array([1.0, 0.0, 0.5])
    vy = np.array([0.0, 2.0, 0.5])
    vz = np.array([0.0, 0.0, -1.0])
    a.recalc(o, o, o, vx, vy, vz)
    for k in range(3):
        tip = a._geom[k, 0, 1]
        d = np.linalg.norm(a._geom[k, 1:, 0] - tip, axis=1)
        assert np.allclose(d, d[0])


def test_arrows_writes_into_matplotlib_buffer(ax):
    """recalc пишет in-place в тот же массив, который читает matplotlib."""
    a = plot.Arrows(ax, 2, plot.rgba('cyan'))
    assert np.shares_memory(np.asarray(a._segments3d), a._geom)


def test_arrows_rejects_short_input(ax):
    a = plot.Arrows(ax, 10, plot.rgba('cyan'))
    short = np.zeros(5)
    with pytest.raises(IndexError):
        a.recalc(short, short, short, short, short, short, step=2)


# --- §2.4: репер Картана уже единичный, нормировка не нужна ---------------

def test_cartan_frame_is_unit_norm():
    rng = np.random.default_rng(0)
    phi = rng.uniform(0, 4 * np.pi, 1000)
    theta = rng.uniform(0, 4 * np.pi, 1000)
    t = np.stack((np.cos(phi) * np.cos(theta),
                  np.cos(phi) * np.sin(theta),
                  np.sin(phi)), axis=1)
    assert np.allclose(np.linalg.norm(t, axis=1), 1.0)


# --- lib/modes ------------------------------------------------------------

def test_core_metric_bounds():
    R, r = 2.0, 1.0
    theta = np.linspace(0, 4 * np.pi, 500)
    phi = 0.5 * theta
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    m = core_metric(x, y, R, r)
    assert m.min() >= 0.0 and m.max() <= 1.0


def test_adaptive_indices_are_sorted_and_in_range():
    R, r, n = 2.0, 1.0, 600
    theta = np.linspace(0, 4 * np.pi, n)
    phi = 0.5 * theta
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    q = adaptive_indices(x, y, R, r)
    assert q[0] == 0
    assert np.all(np.diff(q) > 0)
    assert q[-1] < n
    # шаг должен реально меняться: в ядре гуще, на периферии реже
    assert np.diff(q).min() < np.diff(q).max()


def test_alpha_profile_in_unit_range():
    R, r = 2.0, 1.0
    x = np.linspace(-3, 3, 100)
    a = alpha_profile(x, x, R, r, peak=0.9, drop=0.75)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_to_segments_shape_and_continuity():
    t = np.linspace(0, 1, 50)
    seg = to_segments(t, 2 * t, 3 * t)
    assert seg.shape == (49, 2, 3)
    # конец сегмента i совпадает с началом сегмента i+1
    assert np.allclose(seg[:-1, 1], seg[1:, 0])


# --- lib/torus ------------------------------------------------------------

def test_torus_mesh_lies_on_surface():
    R, r, n = 3.0, 1.0, 40
    X, Y, Z = Torus(R=R, r=r).mesh(n)
    # (sqrt(x^2+y^2) - R)^2 + z^2 == r^2
    lhs = (np.hypot(X, Y) - R) ** 2 + Z ** 2
    assert np.allclose(lhs, r ** 2)


# --- §1.6: суб-вихри протона не должны совпадать по позиции ---------------

def test_proton_quark_shifts_are_distinct():
    """u1, u2 и d разнесены по сечению шланга во все моменты цикла."""
    for frame in range(0, 360, 7):
        tp = 2 * np.pi * frame / 360
        shift_d = np.pi / 2 + 0.5 * np.sin(2 * tp)
        recoil = (np.pi / 8) * np.cos(2 * tp)
        shifts = np.array([recoil, np.pi + recoil, shift_d])
        gaps = np.abs(shifts[:, None] - shifts[None, :])
        gaps = np.minimum(gaps, 2 * np.pi - gaps)
        np.fill_diagonal(gaps, np.inf)
        assert gaps.min() > 0.3, f"кадр {frame}: сдвиги {shifts}"


# --- §2.1: alpha_running.py должен ПРЕДСКАЗЫВАТЬ, а не подгоняться --------

# Файл переименован из a1.py; псевдоним a1 оставлен, чтобы не трогать тела тестов.
import alpha_running as a1


def test_alpha_running_calibrated_to_codata_today():
    """Калибровка воспроизводит CODATA ровно при a = 1."""
    assert a1.inv_alpha(1.0) == pytest.approx(a1.INV_ALPHA_CODATA, abs=1e-9)
    assert a1.delta_alpha_over_alpha(1.0) == pytest.approx(0.0, abs=1e-15)


def test_alpha_running_is_not_self_fitting():
    """РЕГРЕССИЯ: старая версия давала CODATA на ЛЮБОЙ эпохе.

    Если этот тест падает, значит в модель вернулся свободный параметр,
    решаемый из ответа, и предсказания снова нет.
    """
    for z in (0.14, 2.4, 1100.0, 1e12):
        a = 1.0 / (1.0 + z)
        assert abs(a1.inv_alpha(a) - a1.INV_ALPHA_CODATA) > 1e-6, (
            f"1/alpha совпал с CODATA при z={z} — модель снова подгоняется")


def test_alpha_running_prediction_independent_of_calibration():
    """delta_inv_alpha не зависит ни от калибровки, ни от статических членов.

    Именно это делает форму зависимости от эпохи предсказанием: сдвиг
    свободного параметра её не меняет. (Сама Delta alpha/alpha зависит от
    калибровки через общую нормировку 1/alpha ~ 137 в знаменателе — это
    не подгонка формы, а выбор единиц.)
    """
    a = 1.0 / (1.0 + 2.4)
    base = a1.delta_inv_alpha(a)
    saved_c, saved_s = a1.C_CALIB, a1.STATIC_TERM
    try:
        a1.C_CALIB = saved_c + 17.0        # произвольные сдвиги
        a1.STATIC_TERM = saved_s - 4.5
        assert a1.delta_inv_alpha(a) == pytest.approx(base, rel=1e-12)
    finally:
        a1.C_CALIB, a1.STATIC_TERM = saved_c, saved_s


def test_alpha_running_koide_matches_brute_force():
    """Аналитическая подгонка Койде совпадает с прежним перебором."""
    def brute(m1, m2, m3, steps=200000):
        sq = [np.sqrt(m) for m in (m1, m2, m3)]
        x = np.array(sq) / sum(sq)
        phi = np.linspace(0, 2 * np.pi, steps, endpoint=False)
        ideal = np.stack([np.cos(phi + 2 * np.pi * i / 3) for i in range(3)])
        ideal = ideal / np.linalg.norm(ideal, axis=0)
        err = ((x[:, None] - ideal) ** 2).sum(axis=0)
        return np.sqrt(err.min())

    for ms in ((0.510998, 105.65837, 1776.86),
               (2.16, 1270.0, 172760.0),
               (4.67, 93.4, 4180.0)):
        assert a1.koide_fit(*ms)[0] == pytest.approx(brute(*ms), abs=1e-7)


def test_alpha_running_ir_term_is_linear_in_log_a():
    """ИК-добег линеен по ln(a) — на этом держится вывод про дрейф."""
    slope = a1.C_COSMO * np.sqrt(3.0) * np.tan(a1.THETA_W)
    for a in (1.0, 0.5, 1e-6):
        d = a1.ir_horizon_term(a) - a1.ir_horizon_term(1.0)
        assert d == pytest.approx(slope * np.log(a), rel=1e-12)


def test_alpha_running_qcd_term_scales_as_k_squared():
    for a in (1.0, 1e-11):
        base = a1.qcd_thermal_term(a, 1.0)
        assert a1.qcd_thermal_term(a, 3.0) == pytest.approx(9.0 * base, rel=1e-12)


def test_alpha_running_qcd_term_melts_at_high_temperature():
    """При T >> T_c конденсат экранирован, вклад стремится к нулю."""
    assert a1.qcd_thermal_term(1.0) > 1.0                  # сегодня работает
    assert a1.qcd_thermal_term(1e-13) < 1e-6               # в КХД-плазме погас


def test_alpha_running_prediction_scales_with_ir_coeff():
    """delta_inv_alpha строго линейна по коэффициенту ИК-члена.

    На этом держится расчёт max_allowed_ir_coeff().
    """
    a = 1.0 / (1.0 + 0.14)
    full = a1.delta_inv_alpha(a, ir_coeff=1.0)
    assert a1.delta_inv_alpha(a, ir_coeff=0.5) == pytest.approx(0.5 * full, rel=1e-12)
    assert a1.delta_inv_alpha(a, ir_coeff=0.0) == pytest.approx(0.0, abs=1e-15)


def test_alpha_running_reports_exclusion():
    """Ожидаемый на сегодня итог: модель исключена низкими z."""
    rows = a1.evaluate()
    by = {r["label"]: r for r in rows}
    assert not by["Оклo (реактор)"]["ok"]
    assert by["Оклo (реактор)"]["ratio"] > 100
    assert a1.max_allowed_ir_coeff(rows) < 1.0


# --- §2.2 photon_broadening.py --------------------------------------------

import photon_broadening as pb


def test_pb_plasma_frequency_is_physical():
    """omega_p IGM должна быть ~10 рад/с, а не 1e-13 (старое значение)."""
    assert 1.0 < pb.OMEGA_P0 < 100.0
    assert 1e-8 < pb.N_E0 * 1e-6 < 1e-6      # см^-3, порядок IGM


def test_pb_broadening_has_the_square():
    """tau ~ |GDD|/tau_cosmo в асимптотике — это возможно только с квадратом.

    Без квадрата (старая ошибка) зависимость была бы sqrt(|GDD|)/tau.
    """
    w = 2 * np.pi * pb.C_LIGHT / 1e-6
    tau0, z = 1e-15, 1.0
    tc, tt = pb.durations(z, tau0, w)
    g = abs(pb.gdd(z, w))
    assert g / tc ** 2 > 100                  # глубоко в асимптотике
    assert tt == pytest.approx(g / tc, rel=1e-3)


def test_pb_no_nan_for_negative_beta2():
    """beta_2 < 0 не должна давать nan (старая версия могла)."""
    w = 2 * np.pi * pb.C_LIGHT / 1e-6
    assert pb.gdd(1.0, w) < 0
    for z in (0.0, 0.01, 1.0, 10.0):
        tc, tt = pb.durations(z, 1e-15, w)
        assert np.isfinite(tt) and tt >= tc


def test_pb_gdd_reduces_to_flat_space_at_low_z():
    """При z << 1 интеграл должен переходить в -omega_p^2 * L / (c * omega^3)."""
    w = 2 * np.pi * pb.C_LIGHT / 1e-6
    z = 1e-4
    L = pb.comoving_distance(z)
    flat = -(pb.OMEGA_P0 ** 2) * L / (pb.C_LIGHT * w ** 3)
    assert pb.gdd(z, w) == pytest.approx(flat, rel=1e-3)


def test_pb_distance_redshift_are_consistent():
    """1 млрд св. лет — это z ~ 0.07, а не z = 1 (старое противоречие)."""
    d_at_z1 = pb.comoving_distance(1.0) / pb.LIGHT_YEAR / 1e9
    assert 10.0 < d_at_z1 < 12.0
    z_1gly = 0.0701
    d = pb.comoving_distance(z_1gly) / pb.LIGHT_YEAR / 1e9
    assert d == pytest.approx(1.0, rel=0.02)


def test_pb_bandwidth_check_flags_femtosecond_pulse():
    """Импульс 1 фс на 1 мкм слишком широкополосен для разложения до beta_2."""
    w = 2 * np.pi * pb.C_LIGHT / 1e-6
    frac, _ = pb.bandwidth_check(1e-15, w)
    assert frac > 1.0
    frac_ps, _ = pb.bandwidth_check(1e-12, w)
    assert frac_ps < 0.01


# --- §2.3 bell.py: CHSH ---------------------------------------------------

import bell


@pytest.fixture(scope="module")
def phase():
    return np.random.default_rng(bell.SEED).uniform(0, 4 * np.pi, 50000)


def test_bell_outcomes_are_pm_one(phase):
    o = bell.outcomes(phase, (0.0, 1.0, 2.0))
    assert set(np.unique(o)).issubset({-1.0, 1.0})


def test_bell_chsh_does_not_violate_classical_bound(phase):
    s, sem = bell.chsh(phase, bell.CANONICAL)
    assert abs(s) <= bell.CLASSICAL_BOUND + 1e-9
    assert abs(s) == pytest.approx(bell.CLASSICAL_BOUND, abs=1e-6)


def test_bell_chsh_is_pm2_per_sample(phase):
    """Ключевая алгебра: A(B-B') + A'(B+B') = +-2 на каждой реализации."""
    rng = np.random.default_rng(3)
    for _ in range(5):
        angles = tuple(rng.uniform(0, 4 * np.pi, 4))
        _s, _sem, per = bell.chsh(phase, angles, return_samples=True)
        assert set(np.unique(per)).issubset({-2.0, 2.0})


def test_bell_scan_never_exceeds_two(phase):
    """Максимум |S| по сетке углов не превышает классический предел."""
    best, _angles = bell.scan_max_chsh(phase, n_angles=24)
    assert best <= bell.CLASSICAL_BOUND + 1e-6
    assert best < bell.TSIRELSON


def test_bell_correlation_tracks_cosine(phase):
    """Модель повторяет форму cos(x), но не совпадает с ней точно."""
    x = np.linspace(0, 4 * np.pi, 200)
    model = bell.correlation_curve(phase, x)
    err = np.abs(model - np.cos(x))
    assert err.max() < 0.15          # форма та же
    assert err.max() > 0.02          # но систематическое отклонение есть
