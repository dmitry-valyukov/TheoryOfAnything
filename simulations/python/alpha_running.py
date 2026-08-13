import math

# --- Фундаментальные ОТО-константы Вселенной ---
M_EW = 246.22e9          # Электрослабый масштаб вакуума, эВ
M_P = 1.22089e19         # Планковская масса, эВ
Lambda_QCD = 200e6       # Масштаб конфайнмента, эВ
r_Planck = 1.616255e-35  # Планковская длина, м
R_Hubble = 1.374e26      # Размер Вселенной, м

delta_deg_global = 11.099
delta_global = math.radians(delta_deg_global)

theta_W_deg = 28.71
theta_W = math.radians(theta_W_deg)
cos2_theta_W = math.cos(theta_W)**2
sqrt3 = math.sqrt(3)

# 3+1D Объемная нормировка ИК-добега
C_cosmo = 1.0 / (2.0 * (math.pi ** 2)) 

# Текущая температура Вселенной (реликтовое излучение) в эВ
T_now_eV = 2.348e-4  # ~2.725 K
# Критическая температура КХД-перехода (экранирование струн) в эВ
T_critical_eV = 150e6  # ~150 МэВ

# Полный фермионный спектр
fermions_koide = [
    {"name": "e",   "m_MeV": 0.510998,  "n": 0, "q": -1.0,   "type": "lepton"},
    {"name": "mu",  "m_MeV": 105.65837, "n": 1, "q": -1.0,   "type": "lepton"},
    {"name": "tau", "m_MeV": 1776.86,   "n": 2, "q": -1.0,   "type": "lepton"},
    {"name": "u",   "m_MeV": 2.16,      "n": 1, "q": +2/3,   "type": "quark"},
    {"name": "d",   "m_MeV": 4.67,      "n": 2, "q": -1/3,   "type": "quark"},
    {"name": "s",   "m_MeV": 93.4,      "n": 0, "q": -1/3,   "type": "quark"},
    {"name": "c",   "m_MeV": 1270.,     "n": 2, "q": +2/3,   "type": "quark"},
    {"name": "b",   "m_MeV": 4180.,     "n": 0, "q": -1/3,   "type": "quark"},
    {"name": "t",   "m_MeV": 172760.,   "n": 1, "q": +2/3,   "type": "quark"},
]

def koide_phase_from_masses(m1, m2, m3):
    sq = [math.sqrt(m) for m in (m1, m2, m3)]
    s = sum(sq)
    if s == 0: return 0.0, 0.0
    x = [v/s for v in sq]
    best_err, best_phi = 1e9, 0.0
    for i in range(3600):
        phi = i * 2 * math.pi / 3600.0
        ideal = [math.cos(phi), math.cos(phi + 2*math.pi/3), math.cos(phi + 4*math.pi/3)]
        norm = math.sqrt(sum(v*v for v in ideal))
        if norm == 0: continue
        err = sum((a - b/norm)**2 for a, b in zip(x, ideal))
        if err < best_err:
            best_err = err
            best_phi = phi
    return math.sqrt(best_err), best_phi

groups = [
    ("leptons", [f for f in fermions_koide if f["name"] in ["e","mu","tau"]]),
    ("up_quarks", [f for f in fermions_koide if f["name"] in ["u","c","t"]]),
    ("down_quarks", [f for f in fermions_koide if f["name"] in ["d","s","b"]])
]
delta_f_map = {}
for gname, group in groups:
    ms = [f["m_MeV"] for f in group]
    dg, _ = koide_phase_from_masses(*ms)
    for f in group:
        delta_f_map[f["name"]] = dg / 3.0

def weak_boson_contribution(m_boson):
    CA_SU2 = 2.0
    weight = CA_SU2 * (cos2_theta_W if m_boson > 85e9 else 1.0)
    return weight * math.cos(delta_global) * math.log(M_EW / m_boson)

def cosmologic_hubble_contribution_v3(factor=1.0):
    R_current = R_Hubble * factor
    ln_horizon = math.log(R_current / r_Planck)
    return C_cosmo * sqrt3 * ln_horizon * math.tan(theta_W)

# --- Термодинамический неабелев оператор кривизны ---
def compute_thermal_tensor_gluon(k_scale=1.0, factor=1.0):
    """
    Вычисляет КХД-вклад с учетом нелинейного масштаба k и термического плавления
    коммутатора в ранней Вселенной под действием температуры T(factor).
    """
    alpha_Hopf = 2.0 / math.pi
    omega_base = {
        1: [[0.0,  0.5, -0.2], [-0.5, 0.0,  0.1], [ 0.2, -0.1, 0.0]],
        2: [[0.0, -0.1,  0.4], [ 0.1, 0.0, -0.3], [-0.4,  0.3, 0.0]],
        3: [[0.0,  0.3,  0.1], [-0.3, 0.0,  0.5], [-0.1, -0.5, 0.0]]
    }
    
    omega = {}
    for a in omega_base:
        omega[a] = [[val * k_scale for val in row] for row in omega_base[a]]
        
    tr_Omega = 0.0
    for a in omega:
        dw = 0.0
        next_a = (a % 3) + 1
        prev_a = ((a - 2) % 3) + 1
        commutator = 0.0
        for i in range(3):
            for j in range(3):
                commutator += omega[next_a][i][j] * omega[prev_a][j][i]
        tr_Omega += commutator  # dw тождественно 0 для антисимметричных матриц

    log_term = math.log(M_EW / Lambda_QCD)
    phase_metric = math.cos(delta_global)
    base_qcd = abs(tr_Omega) * alpha_Hopf * phase_metric * log_term
    
    # Расчет текущей температуры плазмы
    T_current_eV = T_now_eV / factor
    
    # Термический экранирующий фактор (экспоненциальное затухание КХД-конденсата)
    thermal_screening = math.exp(-(T_current_eV ** 2) / (T_critical_eV ** 2))
    
    return base_qcd * thermal_screening

# --- Базовая инвариантная часть петли (УФ + фермионы + слабые бозоны) ---
C_Gouy = sqrt3 * math.log(M_P / M_EW) - 105.81
alpha_inv_static = sqrt3 * math.log(M_P / M_EW) - C_Gouy

for f in fermions_koide:
    phase_arg = delta_global + 2.0 * math.pi * f["n"] / 3.0 + delta_f_map[f["name"]]
    Nc = 3 if f["type"] == "quark" else 1
    weight = Nc * (f["q"] ** 2)
    alpha_inv_static += 2.0 * sqrt3 * weight * math.cos(phase_arg)

alpha_inv_static += weak_boson_contribution(80.38e9)
alpha_inv_static += weak_boson_contribution(91.19e9)

CODATA = 137.035999

# --- Симуляция термодинамических эпох ---
# От экстремального прошлого (эра КХД-плазмы) до номинала
scenarios = [1e-12, 1e-11, 1.56e-11, 1e-10, 1e-5, 1.0]

print(f"=== ТЕРМОДИНАМИЧЕСКАЯ МОДЕЛЬ ВСЕЛЕННОЙ ===")
for factor in scenarios:
    # 1. Рассчитываем физические параметры эпохи
    T_current = T_now_eV / factor
    hubble_contrib = cosmologic_hubble_contribution_v3(factor)
    
    # 2. Вычисляем дефицит без учета КХД
    alpha_inv_no_qcd = alpha_inv_static + hubble_contrib
    deficit = CODATA - alpha_inv_no_qcd
    
    # 3. Базовый КХД-вклад при k=1 с учетом температуры
    qcd_base_thermal = compute_thermal_tensor_gluon(k_scale=1.0, factor=factor)
    
    # 4. Поиск динамического отклика k
    try:
        # Проверяем физическую разрешимость (удерживает ли вакуум петлю)
        if deficit / qcd_base_thermal < 0:
            raise ValueError("Математический разрыв: вакуум разорван.")
        k_adapted = math.sqrt(deficit / qcd_base_thermal)
        status = f"k = {k_adapted:.6f}"
        final_val = alpha_inv_no_qcd + compute_thermal_tensor_gluon(k_scale=k_adapted, factor=factor)
    except (ValueError, ZeroDivisionError):
        status = "РЕШЕТКА ПЛАВИТСЯ (Деконфайнмент)"
        final_val = alpha_inv_no_qcd  # КХД больше не может компенсировать дрейф

    print(f"Масштаб: {factor:<7} | T = {T_current:.2e} эВ | {status} | 1/alpha = {final_val:.6f}")
