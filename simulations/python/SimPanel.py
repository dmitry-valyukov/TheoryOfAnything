"""SimPanel.py — панель запуска симуляций и расчётов.

Сканирует *.py в своей папке БЕЗ ИМПОРТА (через ast — ни один скрипт при
сканировании не исполняется), извлекает метаданные и группирует по главам
теории. Метаданные берутся из того, что в скриптах уже есть:

    описание   — первый абзац докстринга модуля (или шапка из #-комментариев);
    вид        — определяется по коду: анимация (FuncAnimation/plot.animate),
                 рендер PNG (matplotlib.use("Agg") + savefig), графика
                 (plt.show), иначе текстовый расчёт;
    группа     — по ссылкам на главы в докстринге: «подглава 2.8», «[T-42]»,
                 «theory/02-...»; побеждает самая частая глава; если ссылок
                 нет — по ключевым словам имени файла;
    запуск     — блок «Запуск:» из докстринга, если он есть.

Явное переопределение (по желанию) — словарь в начале скрипта:

    PANEL = {"group": 5, "kind": "анимация", "title": "..."}

Интерфейс — tkinter (стандартная библиотека, ничего ставить не надо):
слева дерево по группам, справа карточка скрипта, поле аргументов и две
кнопки: «Запустить отдельно» (своё окно/консоль) и «Вывод в панель»
(для текстовых расчётов — стриминг stdout в нижнее окно).

Настройка. При первом запуске рядом создаётся simpanel.json — в нём
шрифты, размеры и цвета; правится любым редактором. Поле "theme" выбирает
пресет ("light" | "dark"), словарь "colors" перекрывает отдельные цвета
пресета. Размер шрифта меняется и на лету: Ctrl+колесо мыши или
Ctrl+«+»/«-» — и запоминается в том же файле.

Запуск:  python SimPanel.py            — панель
         python SimPanel.py --list     — то же сканирование текстом, без GUI
         python SimPanel.py --config   — путь и текущие настройки
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))

# названия — по фактическим главам theory/NN-*.md
GROUPS = {
    1: "Глава 1 · Упругий вакуум",
    2: "Глава 2 · Топология и масса",
    3: "Глава 3 · Электродинамика",
    4: "Глава 4 · Заряд и лептоны",
    5: "Глава 5 · Цвет и адроны",
    6: "Глава 6 · Конфайнмент и квантование",
    7: "Глава 7 · Гравитация",
    8: "Глава 8 · Чёрные дыры",
    9: "Глава 9 · Нелокальность",
    10: "Глава 10 · Верификация",
    0: "Без привязки",
}

# запасная привязка по имени файла, когда в докстринге нет ссылок на главы
NAME_HINTS = (
    (r"proton|neutron|baryon|ddd|quark|color", 5),
    (r"photon", 3),
    (r"alpha", 4),
    (r"bell|chsh", 4),
    (r"electron|spinor|torus|soliton|cubicquintic|radial|hopf", 2),
    (r"vacuum|medium|dirac|jamming|cube", 1),
    (r"grav", 7),
    (r"weak", 4),
    (r"lepton|spectrum", 4),
    (r"maxwell", 3),
)

EXCLUDE = re.compile(r"^(_|test_|SimPanel)|^(bootstrap|setup)", re.I)


# ---------------------------------------------------------------------------
# Сканер
# ---------------------------------------------------------------------------

def leading_comments(src: str) -> str:
    """Шапка из #-комментариев для скриптов без докстринга."""
    out = []
    for line in src.splitlines():
        t = line.strip()
        if t.startswith("#"):
            out.append(t.lstrip("# ").rstrip())
        elif t and not t.startswith(("import", "from", '"', "'")):
            break
        elif t.startswith(("import", "from")) and out:
            break
    return "\n".join(out)


def detect_kind(src: str) -> str:
    if re.search(r"FuncAnimation|plot\.animate", src):
        return "анимация"
    if re.search(r'matplotlib\.use\(["\']Agg', src) and "savefig" in src:
        return "рендер PNG"
    if re.search(r"plt\.show|plot\.init", src):
        return "графика"
    return "расчёт (текст)"


CHAP_PATTERNS = (
    re.compile(r"theory/(\d{1,2})-"),
    re.compile(r"подглав\w*\s+(\d{1,2})\."),
    re.compile(r"главам?\s+(\d{1,2})\b"),
    re.compile(r"\[(\d{1,2})\.\d"),
    # ссылка вида (2.8) или (1.5.7): ровно секционный формат, чтобы не
    # путать с числовыми значениями вроде (2.828)
    re.compile(r"\((\d{1,2})\.\d{1,2}(?:\.\d{1,2})?\)"),
)


def detect_chapters(doc: str) -> list[int]:
    votes: dict[int, int] = {}
    for pat in CHAP_PATTERNS:
        for m in pat.finditer(doc):
            ch = int(m.group(1))
            if 1 <= ch <= 10:
                votes[ch] = votes.get(ch, 0) + 1
    return sorted(votes, key=lambda c: -votes[c])


def parse_usage(doc: str) -> list[str]:
    lines, grab = [], False
    for line in doc.splitlines():
        if re.match(r"\s*Запуск\s*:", line):
            grab = True
            tail = line.split(":", 1)[1].strip()
            if tail:
                lines.append(tail)
            continue
        if grab:
            t = line.strip()
            if not t or (lines and not (t.startswith("python") or "—" in t
                                        or "-" in t[:2] or t[0].islower())):
                if not t:
                    break
            if t:
                lines.append(t)
            else:
                break
    return lines[:6]


def module_panel_dict(tree: ast.Module) -> dict:
    """Явное переопределение PANEL = {...} на уровне модуля."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "PANEL":
                    try:
                        val = ast.literal_eval(node.value)
                        if isinstance(val, dict):
                            return val
                    except (ValueError, SyntaxError):
                        pass
    return {}


def scan_one(path: str) -> dict | None:
    name = os.path.basename(path)
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
        tree = ast.parse(src)
    except SyntaxError:
        return dict(name=name, title=name, desc="(не разбирается: SyntaxError)",
                    kind="?", group=0, chapters=[], trefs=[], usage=[])
    doc = ast.get_docstring(tree) or leading_comments(src)
    override = module_panel_dict(tree)

    first = (doc.splitlines() or [""])[0].strip()
    title = re.sub(r"^\S+\.py\s*[—-]\s*", "", first) or name
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc) if p.strip()]
    desc = "\n\n".join(paras[1:2]) if paras else "(описания нет)"

    chapters = detect_chapters(doc)
    group = chapters[0] if chapters else 0
    if group == 0:
        for pat, g in NAME_HINTS:
            if re.search(pat, name, re.I):
                group = g
                break
    kind = detect_kind(src)
    trefs = sorted({int(m) for m in re.findall(r"T-(\d+)", doc)})

    rec = dict(name=name, title=title, desc=desc, kind=kind, group=group,
               chapters=chapters, trefs=trefs, usage=parse_usage(doc))
    rec.update({k: override[k] for k in ("group", "kind", "title")
                if k in override})
    return rec


def scan_all() -> list[dict]:
    out = []
    for f in sorted(os.listdir(HERE), key=str.lower):
        if not f.endswith(".py") or EXCLUDE.search(f):
            continue
        rec = scan_one(os.path.join(HERE, f))
        if rec:
            out.append(rec)
    return out


def print_list():
    recs = scan_all()
    by_group: dict[int, list[dict]] = {}
    for r in recs:
        by_group.setdefault(r["group"], []).append(r)
    for g in sorted(by_group, key=lambda x: (x == 0, x)):
        print(f"\n{'=' * 74}\n{GROUPS[g]}\n{'=' * 74}")
        for r in by_group[g]:
            t = ",".join(f"T-{x}" for x in r["trefs"][:4])
            print(f"  {r['name']:<26} [{r['kind']:>14}] {t}")
            print(f"      {r['title'][:96]}")


# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

CFG_PATH = os.path.join(HERE, "simpanel.json")

THEMES = {
    "light": dict(bg="#f2f2f2", fg="#111111", card_bg="#fdf9f1",
                  card_fg="#111111", accent="#0040A7",
                  out_bg="#ECF8F7", out_fg="#000000",
                  tree_bg="#ffffff", tree_fg="#111111",
                  sel_bg="#cfe2f3", sel_fg="#111111",
                  border ="#609ad4"),
    "dark": dict(bg="#1e1e1e", fg="#dddddd", card_bg="#252526",
                 card_fg="#dddddd", accent="#66aaff",
                 out_bg="#0c0c0c", out_fg="#cccccc",
                 tree_bg="#252526", tree_fg="#dddddd",
                 sel_bg="#264f78", sel_fg="#ffffff",
                 border ="#264f78"),
}

DEFAULT_CFG = {
    "theme": "light",          # light | dark
    "ttk_theme": "",           # "" = системная; варианты: clam, alt, vista...
    "ui_font": "Segoe UI",
    "ui_size": 11,
    "card_font": "Georgia",
    "card_size": 12,
    "mono_font": "Consolas",
    "mono_size": 11,
    "geometry": "1180x720",
    "colors": {},              # перекрывает цвета темы, ключи как в THEMES
}


def load_cfg() -> dict:
    import json
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CFG_PATH):
        try:
            user = json.load(open(CFG_PATH, encoding="utf-8"))
            cfg.update({k: user[k] for k in DEFAULT_CFG if k in user})
        except (ValueError, OSError) as e:
            print(f"simpanel.json не читается ({e}) — настройки по умолчанию")
    else:
        save_cfg(cfg)
    return cfg


def save_cfg(cfg: dict):
    import json
    try:
        json.dump(cfg, open(CFG_PATH, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    except OSError:
        pass


def palette(cfg: dict) -> dict:
    pal = dict(THEMES.get(cfg.get("theme", "light"), THEMES["light"]))
    pal.update(cfg.get("colors", {}))
    return pal


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def run_gui():
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("tkinter недоступен в этом питоне; текстовый режим: "
              "python SimPanel.py --list")
        sys.exit(1)

    cfg = load_cfg()
    pal = palette(cfg)
    recs = {r["name"]: r for r in scan_all()}

    root = tk.Tk()
    root.title("Симуляции модели — панель запуска")
    root.geometry(cfg.get("geometry", "1180x720"))
    root.configure(background=pal["bg"])

    style = ttk.Style(root)
    if cfg.get("ttk_theme"):
        try:
            style.theme_use(cfg["ttk_theme"])
        except tk.TclError:
            print(f"ttk-темы «{cfg['ttk_theme']}» нет; доступны:",
                  ", ".join(style.theme_names()))
    elif cfg.get("theme") == "dark":
        # системная тема Windows игнорирует цвета Treeview — берём clam
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

    ui_font = lambda sz=0: (cfg["ui_font"], cfg["ui_size"] + sz)
    card_font = lambda sz=0: (cfg["card_font"], cfg["card_size"] + sz)
    mono_font = lambda sz=0: (cfg["mono_font"], cfg["mono_size"] + sz)

    style.configure(".", background=pal["bg"], foreground=pal["fg"], font=ui_font())
    style.configure("Treeview", background=pal["tree_bg"],
                    fieldbackground=pal["tree_bg"],
                    foreground=pal["tree_fg"], font=ui_font(),
                    rowheight=int(cfg["ui_size"] * 2.2))
    style.map("Treeview", background=[("selected", pal["sel_bg"])],
              foreground=[("selected", pal["sel_fg"])])
    style.configure("TButton", font=ui_font())
    style.configure("TLabel", background=pal["bg"], foreground=pal["fg"],
                    font=ui_font())

    pw = ttk.PanedWindow(root, orient="horizontal")
    pw.pack(fill="both", expand=True)

    # --- дерево слева
    left = ttk.Frame(pw)
    tree = ttk.Treeview(left, show="tree", selectmode="browse")
    ys = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=ys.set)
    tree.pack(side="left", fill="both", expand=True)
    ys.pack(side="right", fill="y")
    pw.add(left, weight=1)

    by_group: dict[int, list[dict]] = {}
    for r in recs.values():
        by_group.setdefault(r["group"], []).append(r)
    for g in sorted(by_group, key=lambda x: (x == 0, x)):
        parent = tree.insert("", "end", text=GROUPS[g], open=True)
        for r in by_group[g]:
            mark = {"анимация": "▶", "графика": "▶",
                    "рендер PNG": "🖼"}.get(r["kind"], "≡")
            tree.insert(parent, "end", iid=r["name"],
                        text=f"{mark} {r['name']}")

    # --- правая колонка
    right = ttk.Frame(pw)
    pw.add(right, weight=3)

    card = tk.Text(right, height=14, wrap="word", state="disabled",
                   font=card_font(), background=pal["card_bg"],
                   foreground=pal["card_fg"], insertbackground=pal["card_fg"],
                   relief="flat", highlightthickness=1,
                   highlightbackground=pal["border"], highlightcolor=pal["border"])
    card.pack(fill="x", padx=6, pady=(6, 2))

    bar = ttk.Frame(right)
    bar.pack(fill="x", padx=6, pady=2)
    ttk.Label(bar, text="аргументы:").pack(side="left")
    args_var = tk.StringVar()
    tk.Entry(bar, textvariable=args_var, width=44, font=ui_font(),
             background=pal["card_bg"], foreground=pal["card_fg"],
             insertbackground=pal["card_fg"]).pack(
        side="left", padx=6, fill="x", expand=True)
    btn_run = ttk.Button(bar, text=" Запустить отдельно ")
    btn_run.pack(side="left", padx=3)
    btn_cap = ttk.Button(bar, text=" Вывод в панель ")
    btn_cap.pack(side="left", padx=3)

    #out = tk.Text(right, wrap="word", state="disabled", font=mono_font(),
    #              background=pal["out_bg"], foreground=pal["out_fg"],
    #              insertbackground=pal["out_fg"])
    out = tk.Text(right, wrap="word", state="disabled", font=mono_font(),
                  background=pal["out_bg"], foreground=pal["out_fg"],
                  insertbackground=pal["out_fg"],
                  relief="flat", highlightthickness=1,
                  highlightbackground=pal["border"], highlightcolor=pal["border"])

    out.pack(fill="both", expand=True, padx=6, pady=(2, 6))

    status = tk.StringVar(
        value=f"скриптов: {len(recs)}   ·   шрифт Ctrl+колесо / Ctrl+«+»/«-»"
              f"   ·   настройки: {os.path.basename(CFG_PATH)}")
    ttk.Label(root, textvariable=status, anchor="w").pack(fill="x")

    def selected() -> dict | None:
        sel = tree.selection()
        return recs.get(sel[0]) if sel else None

    def show_card(_evt=None):
        r = selected()
        if not r:
            return
        card.configure(state="normal")
        card.delete("1.0", "end")
        chaps = ", ".join(str(c) for c in r["chapters"]) or "—"
        trefs = ", ".join(f"T-{t}" for t in r["trefs"]) or "—"
        card.insert("end", f"{r['name']}   [{r['kind']}]\n") #,
        card.insert("end", f"главы: {chaps}\nдорожная карта: {trefs}\n\n", ("b",))
        card.insert("end", r["title"] + "\n", ("h",))
        card.insert("end", "\n" + r["desc"] + "\n")
        if r["usage"]:
            card.insert("end", "\nЗапуск:\n  " + "\n  ".join(r["usage"]))
        card.tag_configure("h", font=(cfg["card_font"], cfg["card_size"] + 1, "bold"), foreground=pal["accent"])
        card.tag_configure("h", font=(cfg["card_font"], cfg["card_size"], "bold"))
        card.configure(state="disabled")

    def apply_fonts():
        style.configure(".", font=ui())
        style.configure("Treeview", font=ui_font(),
                        rowheight=int(cfg["ui_size"] * 2.2))
        style.configure("TButton", font=ui_font())
        style.configure("TLabel", font=ui_font())
        card.configure(font=card_font())
        out.configure(font=mono_font())
        show_card()

    def zoom(delta):
        cfg["ui_size"] = max(7, min(20, cfg["ui_size"] + delta))
        cfg["mono_size"] = max(6, min(18, cfg["mono_size"] + delta))
        apply_fonts()
        save_cfg(cfg)

    root.bind("<Control-plus>", lambda e: zoom(+1))
    root.bind("<Control-equal>", lambda e: zoom(+1))
    root.bind("<Control-minus>", lambda e: zoom(-1))
    root.bind("<Control-MouseWheel>",
              lambda e: zoom(+1 if e.delta > 0 else -1))

    def on_close():
        cfg["geometry"] = root.geometry().split("+")[0]
        save_cfg(cfg)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    def append_out(text: str):
        out.configure(state="normal")
        out.insert("end", text)
        out.see("end")
        out.configure(state="disabled")

    def env_utf8():
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def run_detached():
        r = selected()
        if not r:
            return
        cmd = [sys.executable, os.path.join(HERE, r["name"])] \
            + args_var.get().split()
        flags = 0
        if os.name == "nt" and r["kind"] == "расчёт (текст)":
            # текстовому расчёту — собственная консоль, чтобы видеть вывод
            flags = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(cmd, cwd=HERE, creationflags=flags, env=env_utf8())
        status.set(f"запущен отдельно: {r['name']}")

    def run_captured():
        r = selected()
        if not r:
            return
        cmd = [sys.executable, "-u", os.path.join(HERE, r["name"])] \
            + args_var.get().split()
        append_out(f"\n$ {' '.join(cmd[1:])}\n")
        status.set(f"выполняется: {r['name']} ...")

        def worker():
            try:
                p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     encoding="utf-8", errors="replace",
                                     env=env_utf8())
                for line in p.stdout:
                    root.after(0, append_out, line)
                p.wait()
                root.after(0, status.set,
                           f"{r['name']}: завершён, код {p.returncode}")
            except Exception as e:                        # noqa: BLE001
                root.after(0, append_out, f"[ошибка запуска] {e}\n")

        threading.Thread(target=worker, daemon=True).start()

    btn_run.configure(command=run_detached)
    btn_cap.configure(command=run_captured)
    tree.bind("<<TreeviewSelect>>", show_card)

    def default_run(_evt):
        r = selected()
        if r:
            (run_captured if r["kind"] == "расчёт (текст)"
             else run_detached)()

    tree.bind("<Double-1>", default_run)
    root.mainloop()


if __name__ == "__main__":
    if "--list" in sys.argv:
        print_list()
    elif "--config" in sys.argv:
        import json
        print(CFG_PATH)
        print(json.dumps(load_cfg(), ensure_ascii=False, indent=2))
    else:
        run_gui()
