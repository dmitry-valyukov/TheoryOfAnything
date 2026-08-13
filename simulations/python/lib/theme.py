"""Палитра для 2D-графиков.

Проверена валидатором в обоих режимах:
контраст к подложке >= 3:1, CVD dE 24.7 (light) / 26.8 (dark),
normal-vision dE 33.6 / 31.8 при пороге 15.
"""

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "s1": "#2a78d6",   # категориальный слот 1
        "s2": "#eb6834",   # слот 2
        "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "s1": "#3987e5",
        "s2": "#d95926",
        "critical": "#d03b3b",
    },
}


def style_axes(ax, t, xlabel="", ylabel="", title=""):
    """Общая отделка осей: скрытая рамка, приглушённая сетка, цвета темы."""
    ax.set_facecolor(t["surface"])
    if xlabel:
        ax.set_xlabel(xlabel, color=t["muted"], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["muted"], fontsize=10)
    if title:
        ax.set_title(title, color=t["ink"], fontsize=13, pad=14)
    ax.grid(True, which="major", color=t["grid"], linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=t["grid"], linewidth=0.4, alpha=0.5, zorder=0)
    ax.tick_params(colors=t["muted"], labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(t["axis"])


def legend(ax, t, **kwargs):
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("fontsize", 9.5)
    leg = ax.legend(**kwargs)
    for text in leg.get_texts():
        text.set_color(t["ink"])
    return leg
