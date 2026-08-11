import lib.plot as plot
from lib.plot import rgba
import numpy as np

# --- Параметры графической модели фотона ---
num_points = 600
arrow_step = 7
num_frames = 120
FPS = 60

# Количество полных периодов (должно быть четное: 2, 4, 6, 8...)
num_periods = 2

# Радиус фигуры
R = plot.SIZE / 2.0

# Условная амплитуда упругого сдвига вакуума (координаты радужной нити)
R_AMP = R / 6.0

# --- Инициализация 3D сцены ---
ax = plot.init(f"ФОТОН в ОТО: Бегущая винтовая фаза внутри временной моды {num_periods*2}π",
               zoom = 4, elev = 30, azim = -45)

# Бугущая линия "волны"
lc = plot.Segments(ax)

# Центрируем относительно начала координат
x_axis = np.linspace(-R, R, num_points)
zeros = np.zeros(num_points)

# Центральная ось полета
center_line_width = 2
center_line, = ax.plot(x_axis, zeros, zeros, color='white', linestyle='--', alpha=0.5, linewidth=center_line_width)

# Создаем постоянные контейнеры для стрелок
num_arrows = (num_points-1)//arrow_step
arrows_E = plot.Arrows(ax, num_arrows, rgba('yellow', 0.25))
arrows_B = plot.Arrows(ax, num_arrows, rgba('cyan', 0.25))

CONST_PI = np.pi
CONST_2PI = CONST_PI * 2
sqrt22 = np.sqrt(2.0) / 2.0

def envelope():
    """Формирование каплевидной огибающей с поджатием хвоста """
    # Динамическая ширина пакета (Модуляция продольной упругости)
    # sigma_0 — базовый размер, k_skew задает степень деформации капли.
    sigma_0 = 5.0
    k_skew = 0.42

    # Нормируем длину пакета на величину Пи для огибающей
    window_pi = x_axis * (CONST_PI / plot.SIZE)

    # Плавная модуляция ширины пакета в зависимости от того, где мы: на носу или на хвосте
    sigma_deformed = sigma_0 * (1.0 + k_skew * np.tanh(window_pi / sigma_0))

    # Каноническая каплевидная огибающая
    gaussian_envelope = np.exp(-(x_axis ** 2) / (2 * sigma_deformed ** 2))

    # Эмулируем поджатие хвоста
    # tmp пробегает от 0-ля до 1/√2
    tmp = np.linspace(0, sqrt22, num_points)
    tmp2 = 1 - np.sqrt(1 - np.multiply(tmp, tmp))
    max_tmp2 = tmp2[len(tmp2) - 1]

    skew_window_pi = (tmp2 * (1.0/max_tmp2) - 0.5) * CONST_PI
    skew_window_pi[:] = skew_window_pi[::-1]

    #return np.cos(skew_window_pi)
    #return gaussian_envelope
    return np.cos(skew_window_pi) * gaussian_envelope

# Длина волны в единицах измерения фигуры
wave_length = plot.SIZE / num_periods

# Волновое число
k = CONST_2PI / wave_length

# Массив начальных фаз
phase0 = num_periods * k * x_axis

env = envelope() * R_AMP

def get_photon_fields(phase: float) -> tuple[tuple[float, float], tuple[float, float], float]:
    """
    Возвращает вектора электрического и магнитного полей, а так же массив фаз:
    ((y_E, z_E), (y_B, z_B), phase)
    """
    phases = (phase0 + phase) % CONST_2PI

    cs = env * np.cos(phases)
    sn = env * np.sin(phases)

    return (cs, sn), (-sn, cs), phases

dash_len = 3
gap_len = 3
pattern = (dash_len, gap_len)
period = dash_len + gap_len  # период для циклического смещения
# set_linestyle масштабирует размеры линий согласно установленной ранее center_line_width
period_length = wave_length * period * R * center_line_width / CONST_2PI / num_periods

# Приращение фазы на фрейм
dphi = CONST_2PI / num_frames

# Колбэк анимации, передаётся номер фрейма, всего фреймов num_frames
def update(frame: int):
    phase = dphi * frame

    offset = (phase * period_length) % period
    center_line.set_linestyle((offset, pattern))

    # Получаем расчетные поля для текущего кадра
    (y, z), (by, bz), phases = get_photon_fields(phase)

    # Обновление геометрии волнового фронта
    points = np.array([x_axis, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc.set_segments(segments)

    # Цвета нити показывают фазу волны
    lc.set_array(phases[:-1] / CONST_2PI)

    # Начало стрелок E — на центральной оси
    arrows_E.recalc(x_axis, zeros, zeros, zeros, y, z, arrow_step)

    # Стрелки магнитного кручения B растут из самой нити по касательной
    arrows_B.recalc(x_axis, y, z, zeros, by, bz, arrow_step)

    return [center_line, lc, arrows_E, arrows_B]

# Запуск зацикленной анимации
plot.animate(update, num_frames, 1000/FPS)
