"""Выбор интерактивного бэкенда matplotlib с безопасным фолбэком.

Импортировать ПЕРВЫМ, до `matplotlib.pyplot` — иначе бэкенд уже зафиксирован.

    import lib.backend  # noqa: F401
    import matplotlib.pyplot as plt

Приоритет можно переопределить переменной окружения MPLBACKEND —
тогда этот модуль ничего не трогает.
"""

import os

import matplotlib

#: порядок попыток; последний (Agg) всегда доступен и рисует в файл
PREFERRED = ("QtAgg", "TkAgg", "Agg")


def select(preferred=PREFERRED) -> str:
    """Ставит первый доступный бэкенд из списка и возвращает его имя."""
    if os.environ.get("MPLBACKEND"):
        # пользователь задал явно — уважаем выбор
        return matplotlib.get_backend()

    for name in preferred:
        try:
            matplotlib.use(name)
            return name
        except Exception:
            continue
    return matplotlib.get_backend()


BACKEND = select()
