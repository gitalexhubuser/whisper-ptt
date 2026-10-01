# ============ CUDA DLL PATCH ============
import os
import sys

if sys.platform == "win32":
    try:
        import site
        for sp in site.getsitepackages():
            for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin", "nvidia/cuda_runtime/bin", "nvidia/cuda_nvrtc/bin"):
                p = os.path.join(sp, sub.replace("/", os.sep))
                if os.path.isdir(p):
                    os.add_dll_directory(p)
                    os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                    print(f"[cuda] DLL dir: {p}")
    except Exception as e:
        print(f"[cuda] patch failed: {e}")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ============ CTranslate2 shim (ctranslate2 ожидает pkg_resources) ============
if "pkg_resources" not in sys.modules:
    class _PRShim:
        @staticmethod
        def resource_filename(mod, rel=""):
            import importlib.util
            spec = importlib.util.find_spec(mod)
            p = os.path.dirname(spec.origin) if (spec and spec.origin) else ""
            return os.path.join(p, rel)
    sys.modules["pkg_resources"] = _PRShim()

# ============ WinAPI SendInput (Unicode-печать без буфера) ============
import ctypes
import time as _time
from ctypes import wintypes

INPUT_KEYBOARD    = 1
KEYEVENTF_KEYUP   = 0x0002
KEYEVENTF_UNICODE = 0x0004
VK_LCONTROL       = 0xA2
VK_V              = 0x56
VK_LSHIFT         = 0xA0
VK_LALT           = 0xA4
VK_LWIN           = 0x5B

ULONG_PTR = ctypes.c_void_p


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx",          wintypes.LONG),
        ("dy",          wintypes.LONG),
        ("mouseData",   wintypes.DWORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",         wintypes.WORD),
        ("wScan",        wintypes.WORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg",    wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _INPUT_UNION),
    ]


_EXPECTED_SIZE = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
if ctypes.sizeof(_INPUT) != _EXPECTED_SIZE:
    print(f"[sendinput] WARNING: sizeof(INPUT)={ctypes.sizeof(_INPUT)}, expected {_EXPECTED_SIZE}")


def _send_key(vk, up=False):
    inp = _INPUT()
    inp.type = INPUT_KEYBOARD
    inp.u.ki.wVk = vk
    inp.u.ki.wScan = ctypes.windll.user32.MapVirtualKeyW(vk, 0)
    inp.u.ki.dwFlags = KEYEVENTF_KEYUP if up else 0
    inp.u.ki.time = 0
    inp.u.ki.dwExtraInfo = None
    ret = ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
    if ret != 1:
        err = ctypes.windll.kernel32.GetLastError()
        print(f"[sendinput] FAILED vk=0x{vk:02X} up={up} ret={ret} err={err}")
    return ret


def _send_unicode_scan(scan, up=False):
    inp = _INPUT()
    inp.type = INPUT_KEYBOARD
    inp.u.ki.wVk = 0
    inp.u.ki.wScan = scan
    inp.u.ki.dwFlags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
    inp.u.ki.time = 0
    inp.u.ki.dwExtraInfo = None
    ret = ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
    if ret != 1:
        err = ctypes.windll.kernel32.GetLastError()
        print(f"[sendinput] UNICODE FAILED scan=0x{scan:04X} up={up} ret={ret} err={err}")
    return ret


def _send_unicode_char(ch):
    code = ord(ch)
    if code > 0xFFFF:
        code -= 0x10000
        high = 0xD800 + (code >> 10)
        low  = 0xDC00 + (code & 0x3FF)
        _send_unicode_scan(high, False)
        _send_unicode_scan(high, True)
        _send_unicode_scan(low, False)
        _send_unicode_scan(low, True)
    else:
        _send_unicode_scan(code, False)
        _send_unicode_scan(code, True)


def _send_unicode_batch(text):
    """Отправляет весь текст целиком за один системный вызов WinAPI SendInput."""
    inputs = []

    for ch in text:
        code = ord(ch)

        if code > 0xFFFF:
            code -= 0x10000
            high = 0xD800 + (code >> 10)
            low  = 0xDC00 + (code & 0x3FF)

            for scan in (high, low):
                inp_down = _INPUT()
                inp_down.type = INPUT_KEYBOARD
                inp_down.u.ki.wVk = 0
                inp_down.u.ki.wScan = scan
                inp_down.u.ki.dwFlags = KEYEVENTF_UNICODE
                inp_down.u.ki.time = 0
                inp_down.u.ki.dwExtraInfo = None
                inputs.append(inp_down)

                inp_up = _INPUT()
                inp_up.type = INPUT_KEYBOARD
                inp_up.u.ki.wVk = 0
                inp_up.u.ki.wScan = scan
                inp_up.u.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
                inp_up.u.ki.time = 0
                inp_up.u.ki.dwExtraInfo = None
                inputs.append(inp_up)

        else:
            inp_down = _INPUT()
            inp_down.type = INPUT_KEYBOARD
            inp_down.u.ki.wVk = 0
            inp_down.u.ki.wScan = code
            inp_down.u.ki.dwFlags = KEYEVENTF_UNICODE
            inp_down.u.ki.time = 0
            inp_down.u.ki.dwExtraInfo = None
            inputs.append(inp_down)

            inp_up = _INPUT()
            inp_up.type = INPUT_KEYBOARD
            inp_up.u.ki.wVk = 0
            inp_up.u.ki.wScan = code
            inp_up.u.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
            inp_up.u.ki.time = 0
            inp_up.u.ki.dwExtraInfo = None
            inputs.append(inp_up)

    if not inputs:
        return 0

    c_inputs = (_INPUT * len(inputs))(*inputs)
    ret = ctypes.windll.user32.SendInput(
        len(inputs),
        c_inputs,
        ctypes.sizeof(_INPUT)
    )

    if ret != len(inputs):
        err = ctypes.windll.kernel32.GetLastError()
        print(f"[sendinput] BATCH FAILED sent {ret}/{len(inputs)} err={err}")

    return ret


def _type_text(text, char_delay=0.001):
    for ch in text:
        _send_unicode_char(ch)
        if char_delay > 0:
            _time.sleep(char_delay)


def _release_all_modifiers():
    for vk in (VK_LCONTROL, VK_LSHIFT, VK_LALT, VK_LWIN):
        _send_key(vk, up=True)


# =========================================


import configparser
import queue
import threading
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from pynput import keyboard
from pystray import Icon, Menu, MenuItem
from PIL import Image, ImageDraw


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.ini"


DEFAULT_CONFIG = """[whisper]
backend = faster
model = large-v3-turbo
device = cuda
compute_type = int8_float32
language = ru
beam_size = 1

[hotkey]
# односимвольные не используй (f9, f10, scroll_lock, pause и т.п.)
key = f9
# hold  = держать F9 во время записи
# toggle = нажал F9 -> запись началась, нажал F9 ещё раз -> запись остановилась
mode = hold

[audio]
sample_rate = 16000

[text]
add_trailing_space = true

[paste]
# method: unicode (рекомендуется) или clipboard
method = unicode
# char_delay = 0.0 для мгновенной пакетной вставки через SendInput
char_delay = 0.0

[log]
enabled = true
file = transcriptions.log

[sound]
enabled = true
file = ready.mp3
volume = 0.7
"""


def ensure_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG, encoding="utf-8")

    cfg = configparser.ConfigParser(
        inline_comment_prefixes=("#", ";")
    )

    cfg.read(CONFIG_PATH, encoding="utf-8")

    return cfg


cfg = ensure_config()


BACKEND      = cfg.get("whisper", "backend",      fallback="faster").strip().lower()
MODEL_NAME   = cfg.get("whisper", "model",        fallback="large-v3-turbo").strip()
DEVICE       = cfg.get("whisper", "device",       fallback="cuda").strip()
COMPUTE_TYPE = cfg.get("whisper", "compute_type", fallback="int8_float32").strip()
LANGUAGE     = cfg.get("whisper", "language",     fallback="ru").strip() or None
BEAM_SIZE    = cfg.getint("whisper", "beam_size", fallback=1)


HOTKEY_STR = cfg.get(
    "hotkey",
    "key",
    fallback="f9"
).strip().lower()

HOTKEY_MODE = cfg.get(
    "hotkey",
    "mode",
    fallback="hold"
).strip().lower()

if HOTKEY_MODE not in ("hold", "toggle"):
    print(
        f"[hotkey] неизвестный mode={HOTKEY_MODE!r}, "
        f"используется hold"
    )
    HOTKEY_MODE = "hold"


SAMPLE_RATE = cfg.getint(
    "audio",
    "sample_rate",
    fallback=16000
)

ADD_SPACE = cfg.getboolean(
    "text",
    "add_trailing_space",
    fallback=True
)


PASTE_METHOD = cfg.get(
    "paste",
    "method",
    fallback="unicode"
).strip().lower()

CHAR_DELAY = cfg.getfloat(
    "paste",
    "char_delay",
    fallback=0.0
)


LOG_ENABLED = cfg.getboolean(
    "log",
    "enabled",
    fallback=True
)

LOG_FILE = APP_DIR / cfg.get(
    "log",
    "file",
    fallback="transcriptions.log"
).strip()


SOUND_ENABLED = cfg.getboolean(
    "sound",
    "enabled",
    fallback=True
)

SOUND_FILE = APP_DIR / cfg.get(
    "sound",
    "file",
    fallback="ready.mp3"
).strip()

SOUND_VOLUME = cfg.getfloat(
    "sound",
    "volume",
    fallback=0.7
)


# ---------- Hotkey resolve (одна конкретная клавиша) ----------
def resolve_hotkey(name):
    """'f9' → keyboard.Key.f9, 'scroll_lock' → keyboard.Key.scroll_lock, 'a' → KeyCode."""
    name = name.strip().lower()

    if name.startswith("f") and name[1:].isdigit():
        if hasattr(keyboard.Key, name):
            return getattr(keyboard.Key, name)

        raise ValueError(
            f"Неизвестная F-клавиша: {name}"
        )

    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)

    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)

    raise ValueError(
        f"Неизвестная клавиша: {name}"
    )


HOTKEY = resolve_hotkey(HOTKEY_STR)

print(
    f"[hotkey] using: {HOTKEY_STR} "
    f"mode={HOTKEY_MODE} ({HOTKEY})"
)


# ---------- Sound ----------
_sound_obj = None
_sound_ready = False


def load_sound():
    global _sound_obj, _sound_ready

    if not SOUND_ENABLED:
        print("[sound] disabled")
        return

    if not SOUND_FILE.exists():
        print(
            f"[sound] file not found: {SOUND_FILE} — звук отключён"
        )
        return

    try:
        import pygame

        pygame.mixer.pre_init(
            frequency=44100,
            size=-16,
            channels=2,
            buffer=512
        )

        pygame.mixer.init()

        _sound_obj = pygame.mixer.Sound(
            str(SOUND_FILE)
        )

        _sound_obj.set_volume(SOUND_VOLUME)

        _sound_ready = True

        print(
            f"[sound] loaded: {SOUND_FILE.name}"
        )

    except Exception as e:
        print(f"[sound] load failed: {e}")


def play_ready_sound():
    if not _sound_ready or _sound_obj is None:
        return

    try:
        _sound_obj.play()
    except Exception as e:
        print(f"[sound] play failed: {e}")


# ---------- Log & internal buffer ----------
_internal_buffer = deque(maxlen=50)


def log_transcription(text):
    _internal_buffer.append(text)

    if not LOG_ENABLED:
        return

    try:
        ts = _time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        with open(
            LOG_FILE,
            "a",
            encoding="utf-8"
        ) as f:
            f.write(
                f"[{ts}] {text}\n"
            )

    except Exception as e:
        print(
            f"[log] write failed: {e}"
        )


def open_log_file(icon_, item):
    if not LOG_FILE.exists():
        if icon_:
            try:
                icon_.notify(
                    f"Лог ещё пуст: {LOG_FILE.name}",
                    "Whisper PTT"
                )
            except Exception:
                pass
        return

    try:
        os.startfile(str(LOG_FILE))

        print(
            f"[tray] opened log: {LOG_FILE}"
        )

    except Exception as e:
        print(
            f"[tray] open log failed: {e}"
        )


def copy_last_to_clipboard(icon_, item):
    if not _internal_buffer:
        if icon_:
            icon_.notify(
                "Буфер расшифровок пуст",
                "Whisper PTT"
            )
        return

    last = _internal_buffer[-1]

    try:
        import pyperclip

        pyperclip.copy(last)

        print(
            f"[tray] copied last to clipboard: {last!r}"
        )

        if icon_:
            icon_.notify(
                f"Скопировано: {last[:60]}",
                "Whisper PTT"
            )

    except Exception as e:
        print(
            f"[tray] copy failed: {e}"
        )


# ---------- UI окно истории ----------
_history_window = None


def open_history_window(icon_, item):
    global _history_window

    import tkinter as tk

    if _history_window is not None:
        try:
            _history_window.deiconify()
            _history_window.lift()
            _history_window.focus_force()
            return

        except Exception:
            _history_window = None

    root = tk.Tk()

    _history_window = root

    root.title(
        "Whisper PTT — история расшифровок"
    )

    root.geometry("700x450")

    top = tk.Frame(root)
    top.pack(
        fill="x",
        padx=8,
        pady=6
    )

    text_widget = None

    def copy_selected():
        try:
            sel = text_widget.get(
                "sel.first",
                "sel.last"
            )

        except tk.TclError:
            sel = ""

        if sel:
            root.clipboard_clear()
            root.clipboard_append(sel)

            print(
                f"[ui] copied selection: {sel[:60]!r}"
            )

    def copy_all():
        all_text = text_widget.get(
            "1.0",
            "end"
        ).strip()

        root.clipboard_clear()
        root.clipboard_append(all_text)

        print(
            f"[ui] copied all ({len(all_text)} chars)"
        )

    def clear_view():
        text_widget.delete(
            "1.0",
            "end"
        )

    def refresh_view():
        text_widget.delete(
            "1.0",
            "end"
        )

        for i, t in enumerate(
            _internal_buffer,
            1
        ):
            text_widget.insert(
                "end",
                f"{i:>3}. {t}\n"
            )

    tk.Button(
        top,
        text="Копировать выделенное",
        command=copy_selected
    ).pack(
        side="left",
        padx=2
    )

    tk.Button(
        top,
        text="Копировать всё",
        command=copy_all
    ).pack(
        side="left",
        padx=2
    )

    tk.Button(
        top,
        text="Очистить вид",
        command=clear_view
    ).pack(
        side="left",
        padx=2
    )

    tk.Button(
        top,
        text="Обновить",
        command=refresh_view
    ).pack(
        side="left",
        padx=2
    )

    tk.Label(
        top,
        text=(
            f"(буфер: до {_internal_buffer.maxlen} шт., "
            f"в файле: {LOG_FILE.name})"
        )
    ).pack(
        side="right"
    )

    frame = tk.Frame(root)

    frame.pack(
        fill="both",
        expand=True,
        padx=8,
        pady=6
    )

    scrollbar = tk.Scrollbar(frame)

    scrollbar.pack(
        side="right",
        fill="y"
    )

    text_widget = tk.Text(
        frame,
        wrap="word",
        yscrollcommand=scrollbar.set,
        font=("Consolas", 11)
    )

    text_widget.pack(
        side="left",
        fill="both",
        expand=True
    )

    scrollbar.config(
        command=text_widget.yview
    )

    refresh_view()

    text_widget.see("end")

    def on_close():
        global _history_window

        _history_window = None

        root.destroy()

    root.protocol(
        "WM_DELETE_WINDOW",
        on_close
    )

    root.mainloop()


# ---------- State & Queues ----------
model = None
model_lock = threading.Lock()
model_ready = threading.Event()

recording = False
audio_chunks = []

# Pre-roll ring buffer (~250ms audio to never miss the first syllable)
_preroll_buffer = deque(maxlen=2)

stream = None
stream_lock = threading.Lock()

_audio_queue = queue.Queue()

icon = None


COLOR_IDLE = (60, 170, 90)
COLOR_REC  = (220, 60, 60)
COLOR_LOAD = (180, 160, 60)
COLOR_ERR  = (120, 120, 120)


def make_image(color):
    img = Image.new(
        "RGBA",
        (64, 64),
        (0, 0, 0, 0)
    )

    d = ImageDraw.Draw(img)

    d.ellipse(
        (4, 4, 60, 60),
        fill=color
    )

    d.rounded_rectangle(
        (26, 14, 38, 40),
        radius=6,
        fill=(255, 255, 255)
    )

    d.arc(
        (18, 26, 46, 50),
        start=0,
        end=180,
        fill=(255, 255, 255),
        width=3
    )

    d.line(
        (32, 46, 32, 54),
        fill=(255, 255, 255),
        width=3
    )

    return img


def set_color(color):
    if icon is not None:
        try:
            icon.icon = make_image(color)
        except Exception:
            pass


def set_tray_status(status_text=None):
    if icon is not None:
        try:
            if status_text:
                icon.title = (
                    f"Whisper PTT — {status_text} "
                    f"({HOTKEY_STR}, {HOTKEY_MODE})"
                )
            else:
                icon.title = (
                    f"Whisper PTT — ready "
                    f"({HOTKEY_STR}, {HOTKEY_MODE})"
                )

        except Exception:
            pass


# ---------- Model ----------
def load_model():
    global model

    try:
        set_tray_status("загрузка 0%")

        # Перехват прогресса скачивания весов
        try:
            from tqdm.auto import tqdm

            class _TrayDownloadProgress(tqdm):
                _last_pct = -1
                _last_t = 0.0

                def __init__(
                    self,
                    *args,
                    **kwargs
                ):
                    kwargs["disable"] = False

                    super().__init__(
                        *args,
                        **kwargs
                    )

                    self._report()

                def update(self, n=1):
                    super().update(n)
                    self._report()

                def _report(self):
                    if self.total and self.total > 0:
                        pct = max(
                            0,
                            min(
                                100,
                                int(
                                    (self.n / self.total)
                                    * 100
                                )
                            )
                        )

                        now = _time.time()

                        if (
                            pct != _TrayDownloadProgress._last_pct
                            and (
                                now
                                - _TrayDownloadProgress._last_t
                                >= 0.15
                                or pct in (0, 100)
                            )
                        ):
                            _TrayDownloadProgress._last_pct = pct
                            _TrayDownloadProgress._last_t = now

                            set_tray_status(
                                f"скачивание {pct}%"
                            )

            import faster_whisper.utils

            faster_whisper.utils.disabled_tqdm = (
                _TrayDownloadProgress
            )

        except Exception as e:
            print(
                f"[whisper] tqdm patch error: {e}"
            )

        # Монитор чтения весов
        stop_monitor = threading.Event()

        def _monitor_loading():
            try:
                import psutil

                proc = psutil.Process()

                b_start = proc.io_counters().read_bytes

                size_map = {
                    "tiny": 75 * 1024 * 1024,
                    "base": 145 * 1024 * 1024,
                    "small": 485 * 1024 * 1024,
                    "medium": 1500 * 1024 * 1024,
                    "large-v3-turbo": 1600 * 1024 * 1024,
                    "turbo": 1600 * 1024 * 1024,
                    "large-v3": 3100 * 1024 * 1024,
                    "large": 3100 * 1024 * 1024,
                }

                expected = size_map.get(
                    MODEL_NAME.lower(),
                    1600 * 1024 * 1024
                )

                last_pct = 0
                t0 = _time.time()

                while not stop_monitor.wait(0.2):
                    read_b = (
                        proc.io_counters().read_bytes
                        - b_start
                    )

                    if read_b > 0:
                        pct = min(
                            98,
                            int(
                                (read_b / expected)
                                * 100
                            )
                        )
                    else:
                        pct = min(
                            90,
                            int(
                                (_time.time() - t0)
                                * 8
                            )
                        )

                    if pct > last_pct:
                        last_pct = pct

                        set_tray_status(
                            f"загрузка {pct}%"
                        )

            except Exception:
                pass

        monitor_th = threading.Thread(
            target=_monitor_loading,
            daemon=True
        )

        monitor_th.start()

        # Автопоиск локальной папки с моделью
        local_dir = None

        for cand in [
            Path(MODEL_NAME),
            APP_DIR / MODEL_NAME,
            APP_DIR / "models" / MODEL_NAME,
            Path("C:/models") / MODEL_NAME,
        ]:
            if (
                cand.is_dir()
                and (cand / "model.bin").exists()
            ):
                local_dir = str(
                    cand.resolve()
                )
                break

        target_model = (
            local_dir
            if local_dir
            else MODEL_NAME
        )

        try:
            if BACKEND == "faster":
                from faster_whisper import WhisperModel

                if local_dir:
                    print(
                        f"[whisper] offline load from: "
                        f"{local_dir}"
                    )
                else:
                    print(
                        f"[whisper] loading "
                        f"{MODEL_NAME} on "
                        f"{DEVICE} "
                        f"({COMPUTE_TYPE})…"
                    )

                model = WhisperModel(
                    target_model,
                    device=DEVICE,
                    compute_type=COMPUTE_TYPE
                )

            elif BACKEND == "openai":
                import whisper

                model = whisper.load_model(
                    MODEL_NAME,
                    device=DEVICE
                )

            else:
                raise ValueError(
                    f"Unknown backend: {BACKEND}"
                )

        finally:
            stop_monitor.set()
            monitor_th.join(
                timeout=1.0
            )

        model_ready.set()

        set_color(COLOR_IDLE)
        set_tray_status(None)

        print(
            f"[whisper] READY "
            f"({BACKEND}, "
            f"{DEVICE}, "
            f"{COMPUTE_TYPE})"
        )

    except Exception as e:
        print(
            f"[whisper] FAILED: {e}"
        )

        set_color(COLOR_ERR)

        if icon:
            try:
                icon.notify(
                    f"Ошибка загрузки модели: {e}",
                    "Whisper PTT"
                )
            except Exception:
                pass


# ---------- Persistent Audio Stream ----------
def _audio_cb(
    indata,
    frames,
    time_info,
    status
):
    if recording:
        audio_chunks.append(
            indata.copy()
        )
    else:
        _preroll_buffer.append(
            indata.copy()
        )


def init_audio_stream():
    global stream

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=2048,
            callback=_audio_cb,
        )

        stream.start()

        print(
            "[audio] persistent stream active "
            "(zero-latency start enabled)"
        )

    except Exception as e:
        print(
            f"[audio] failed to start "
            f"audio stream: {e}"
        )


def start_recording():
    global recording
    global audio_chunks

    if not model_ready.is_set():
        print(
            "[rec] model not ready"
        )
        return

    with stream_lock:
        if recording:
            return

        # Захватываем pre-roll буфер
        audio_chunks = list(
            _preroll_buffer
        )

        _preroll_buffer.clear()

        recording = True

    set_color(COLOR_REC)

    if icon:
        icon.title = (
            f"Whisper PTT — REC "
            f"({HOTKEY_STR}, {HOTKEY_MODE})"
        )

    print(
        "[rec] started"
    )


def stop_recording():
    global recording

    with stream_lock:
        if not recording:
            return

        recording = False

        chunks = list(
            audio_chunks
        )

    set_color(COLOR_IDLE)

    if icon:
        icon.title = (
            f"Whisper PTT — transcribing… "
            f"({HOTKEY_STR}, {HOTKEY_MODE})"
        )

    print(
        "[rec] stopped, queued for transcription"
    )

    _audio_queue.put(chunks)


def _worker_loop():
    """Фоновый поток для последовательной и быстрой обработки аудио."""
    while True:
        chunks = _audio_queue.get()

        try:
            process_audio(chunks)

        except Exception as e:
            print(
                f"[worker] process error: {e}"
            )

        finally:
            _audio_queue.task_done()


def process_audio(chunks):
    if not chunks:
        return

    try:
        audio = (
            np.concatenate(
                chunks,
                axis=0
            )
            .flatten()
            .astype(np.float32)
        )

    except Exception as e:
        print(
            f"[rec] concat error: {e}"
        )
        return

    # Отсекаем слишком короткие клики
    if len(audio) < SAMPLE_RATE * 0.35:
        print(
            "[rec] too short, skip"
        )

        if icon:
            icon.title = (
                f"Whisper PTT — ready "
                f"({HOTKEY_STR}, {HOTKEY_MODE})"
            )

        return

    t0 = _time.time()

    try:
        with model_lock:

            if BACKEND == "faster":
                segments, _ = model.transcribe(
                    audio,
                    language=LANGUAGE,
                    beam_size=BEAM_SIZE,
                    temperature=0.0,
                    without_timestamps=True,
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 300,
                        "speech_pad_ms": 200,
                    },
                    condition_on_previous_text=False,
                )

                text = "".join(
                    s.text
                    for s in segments
                )

            else:
                result = model.transcribe(
                    audio,
                    language=LANGUAGE,
                    fp16=(DEVICE == "cuda"),
                    condition_on_previous_text=False,
                )

                text = result.get(
                    "text",
                    ""
                )

    except Exception as e:
        print(
            f"[rec] transcribe error: {e}"
        )

        if icon:
            icon.title = (
                f"Whisper PTT — ready "
                f"({HOTKEY_STR}, {HOTKEY_MODE})"
            )

        return

    text = text.strip()

    dur = len(audio) / SAMPLE_RATE

    dt = _time.time() - t0

    print(
        f"[rec] {dur:.1f}s audio -> "
        f"{dt:.2f}s "
        f"(rtf={dt/max(dur, 0.01):.2f}) "
        f"-> {text!r}"
    )

    if text:
        log_transcription(text)

        out = (
            text + " "
            if ADD_SPACE
            else text
        )

        if PASTE_METHOD == "unicode":
            type_text_to_active_window(out)
        else:
            paste_via_clipboard(out)

        play_ready_sound()

    else:
        print(
            "[rec] empty result"
        )

    if icon:
        icon.title = (
            f"Whisper PTT — ready "
            f"({HOTKEY_STR}, {HOTKEY_MODE})"
        )


def type_text_to_active_window(text):
    _release_all_modifiers()

    _time.sleep(0.01)

    t0 = _time.time()

    try:
        if CHAR_DELAY > 0:
            _type_text(
                text,
                char_delay=CHAR_DELAY
            )
        else:
            _send_unicode_batch(text)

        dt = _time.time() - t0

        print(
            f"[type] sent "
            f"{len(text)} chars "
            f"in {dt*1000:.1f}ms"
        )

    except Exception as e:
        print(
            f"[type] failed: {e}"
        )


def paste_via_clipboard(text):
    try:
        import pyperclip

        pyperclip.copy(text)

    except Exception as e:
        print(
            f"[paste] clipboard error: {e}"
        )
        return

    _release_all_modifiers()

    _time.sleep(0.02)

    _send_key(
        VK_LCONTROL,
        up=False
    )

    _send_key(
        VK_V,
        up=False
    )

    _time.sleep(0.01)

    _send_key(
        VK_V,
        up=True
    )

    _send_key(
        VK_LCONTROL,
        up=True
    )

    print(
        f"[paste] pasted "
        f"{len(text)} chars "
        f"via clipboard"
    )


# ---------- Listener ----------
def _is_hotkey(key):
    """Сравнение с учётом KeyCode (char) и Key (enum)."""

    if key == HOTKEY:
        return True

    if (
        isinstance(HOTKEY, keyboard.KeyCode)
        and isinstance(key, keyboard.KeyCode)
    ):
        return HOTKEY.char == key.char

    return False


def on_press(key):
    if not _is_hotkey(key):
        return

    # =========================================================
    # TOGGLE MODE
    # Нажал F9 один раз  -> начать запись
    # Нажал F9 второй раз -> остановить запись
    # Отпускание F9 игнорируется
    # =========================================================
    if HOTKEY_MODE == "toggle":

        if recording:
            stop_recording()
        else:
            start_recording()

        return

    # =========================================================
    # HOLD MODE
    # Старое поведение:
    # зажал F9 -> запись
    # отпустил F9 -> остановка
    # =========================================================

    if recording:
        return  # автоповтор F9 — игнор

    start_recording()


def on_release(key):
    if not _is_hotkey(key):
        return

    # В toggle-режиме отпускание F9
    # вообще ничего не делает.
    if HOTKEY_MODE == "toggle":
        return

    # В hold-режиме отпускание F9
    # останавливает запись.
    if not recording:
        return

    stop_recording()


def run_listener():
    with keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    ) as l:
        l.join()


# ---------- Tray ----------
def on_quit(icon_, item):
    global stream

    if stream is not None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    try:
        if icon_:
            icon_.stop()
    finally:
        os._exit(0)


def main():
    global icon

    icon = Icon(
        "whisper-ptt",
        make_image(COLOR_LOAD),

        title=(
            f"Whisper PTT — загрузка 0% "
            f"({HOTKEY_STR}, {HOTKEY_MODE})"
        ),

        menu=Menu(

            MenuItem(
                f"Хоткей: {HOTKEY_STR}",
                None,
                enabled=False
            ),

            MenuItem(
                f"Режим: {HOTKEY_MODE}",
                None,
                enabled=False
            ),

            MenuItem(
                f"Модель: {MODEL_NAME}",
                None,
                enabled=False
            ),

            MenuItem(
                f"Устройство: "
                f"{DEVICE} / {COMPUTE_TYPE}",
                None,
                enabled=False
            ),

            MenuItem(
                f"Метод вставки: "
                f"{PASTE_METHOD}",
                None,
                enabled=False
            ),

            Menu.SEPARATOR,

            MenuItem(
                "Открыть историю расшифровок",
                open_history_window,
                default=True
            ),

            MenuItem(
                "Открыть лог-файл",
                open_log_file
            ),

            MenuItem(
                "Скопировать последнюю в буфер",
                copy_last_to_clipboard
            ),

            Menu.SEPARATOR,

            MenuItem(
                "Выход",
                on_quit
            ),
        ),
    )

    init_audio_stream()

    threading.Thread(
        target=_worker_loop,
        daemon=True
    ).start()

    threading.Thread(
        target=load_sound,
        daemon=True
    ).start()

    threading.Thread(
        target=load_model,
        daemon=True
    ).start()

    threading.Thread(
        target=run_listener,
        daemon=True
    ).start()

    icon.run()


if __name__ == "__main__":
    main()