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

# ============ NO INTERNET ============
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
# Файлы звуков (MP3/WAV). Если файл не найден — используется системный beep
start_file = start.mp3
end_file = end.mp3
ready_file = ready.mp3
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


BACKEND      = cfg.get("whisper", "backend", fallback="faster").strip().lower()
MODEL_NAME   = cfg.get("whisper", "model", fallback="large-v3-turbo").strip()
DEVICE       = cfg.get("whisper", "device", fallback="cuda").strip()
COMPUTE_TYPE = cfg.get("whisper", "compute_type", fallback="int8_float32").strip()
LANGUAGE     = cfg.get("whisper", "language", fallback="ru").strip() or None
BEAM_SIZE    = cfg.getint("whisper", "beam_size", fallback=1)

HOTKEY_STR = cfg.get("hotkey", "key", fallback="f9").strip().lower()
HOTKEY_MODE = cfg.get("hotkey", "mode", fallback="hold").strip().lower()

if HOTKEY_MODE not in ("hold", "toggle"):
    print(f"[hotkey] неизвестный mode={HOTKEY_MODE!r}, используется hold")
    HOTKEY_MODE = "hold"

SAMPLE_RATE = cfg.getint("audio", "sample_rate", fallback=16000)
ADD_SPACE = cfg.getboolean("text", "add_trailing_space", fallback=True)

PASTE_METHOD = cfg.get("paste", "method", fallback="unicode").strip().lower()
CHAR_DELAY = cfg.getfloat("paste", "char_delay", fallback=0.0)

LOG_ENABLED = cfg.getboolean("log", "enabled", fallback=True)
LOG_FILE = APP_DIR / cfg.get("log", "file", fallback="transcriptions.log").strip()

SOUND_ENABLED = cfg.getboolean("sound", "enabled", fallback=True)
SOUND_VOLUME = cfg.getfloat("sound", "volume", fallback=0.7)

# Backward compat: если ready_file нет в конфиге — берём старый file
_ready_fallback = cfg.get("sound", "file", fallback="ready.mp3").strip()
SOUND_START_FILE = cfg.get("sound", "start_file", fallback="start.mp3").strip()
SOUND_END_FILE = cfg.get("sound", "end_file", fallback="end.mp3").strip()
SOUND_READY_FILE = cfg.get("sound", "ready_file", fallback=_ready_fallback).strip()


# ---------- Hotkey resolve ----------
def resolve_hotkey(name):
    """'f9' → keyboard.Key.f9, 'scroll_lock' → keyboard.Key.scroll_lock, 'a' → KeyCode."""
    name = name.strip().lower()

    if name.startswith("f") and name[1:].isdigit():
        if hasattr(keyboard.Key, name):
            return getattr(keyboard.Key, name)
        raise ValueError(f"Неизвестная F-клавиша: {name}")

    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)

    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)

    raise ValueError(f"Неизвестная клавиша: {name}")


HOTKEY = resolve_hotkey(HOTKEY_STR)

print(f"[hotkey] using: {HOTKEY_STR} mode={HOTKEY_MODE} ({HOTKEY})")


# ---------- Sound (winmm.dll — без pygame) ----------
_winmm = ctypes.windll.winmm
_winmm.mciSendStringW.argtypes = [
    ctypes.c_wchar_p,
    ctypes.c_wchar_p,
    ctypes.c_uint,
    ctypes.c_void_p,
]
_winmm.mciSendStringW.restype = ctypes.c_uint

_mci_lock = threading.Lock()
_mci_counter = 0

_sound_start_file = None
_sound_end_file = None
_sound_ready_file = None


def _play_audio_file(filepath):
    """Воспроизводит аудиофайл (MP3/WAV) через Windows winmm (mciSendString).

    Использует уникальный alias для каждого вызова, чтобы звуки не
    прерывали друг друга. Устройство закрывается через 5 секунд в
    фоновом потоке.
    """
    global _mci_counter

    filepath = os.path.abspath(str(filepath))

    with _mci_lock:
        _mci_counter += 1
        alias = f"wpsnd_{_mci_counter}"

        try:
            cmd = f'open "{filepath}" alias {alias}'
            ret = _winmm.mciSendStringW(cmd, None, 0, None)

            if ret != 0:
                # Пробуем с явным указанием типа mpegvideo (для MP3)
                cmd = f'open "{filepath}" type mpegvideo alias {alias}'
                ret = _winmm.mciSendStringW(cmd, None, 0, None)

                if ret != 0:
                    return False

            # Громкость (0..1000)
            try:
                vol = max(0, min(1000, int(SOUND_VOLUME * 1000)))
                _winmm.mciSendStringW(
                    f'setaudio {alias} volume to {vol}',
                    None, 0, None
                )
            except Exception:
                pass

            _winmm.mciSendStringW(f'play {alias}', None, 0, None)

        except Exception as e:
            print(f"[sound] play error: {e}")
            return False

    # Закрытие устройства через 5 секунд
    def _cleanup():
        _time.sleep(5)
        with _mci_lock:
            try:
                _winmm.mciSendStringW(f'close {alias}', None, 0, None)
            except Exception:
                pass

    threading.Thread(target=_cleanup, daemon=True).start()

    return True


def _beep_async(freq, duration):
    """Системный beep в фоновом потоке (не блокирует listener)."""
    threading.Thread(
        target=_try_beep,
        args=(freq, duration),
        daemon=True
    ).start()


def _try_beep(freq, duration):
    try:
        import winsound
        winsound.Beep(freq, duration)
    except Exception:
        pass


def load_sound():
    """Проверяет наличие звуковых файлов. Не требует pygame."""
    global _sound_start_file, _sound_end_file, _sound_ready_file

    if not SOUND_ENABLED:
        print("[sound] disabled")
        return

    for name, attr, filename in [
        ("start", "_sound_start_file", SOUND_START_FILE),
        ("end",   "_sound_end_file",   SOUND_END_FILE),
        ("ready", "_sound_ready_file",  SOUND_READY_FILE),
    ]:
        path = APP_DIR / filename
        if path.exists():
            globals()[attr] = path
            print(f"[sound] {name}: {filename}")
        else:
            print(f"[sound] {name}: {filename} не найден — beep")


def play_start_sound():
    """Звук начала записи."""
    if not SOUND_ENABLED:
        return
    if _sound_start_file and _play_audio_file(str(_sound_start_file)):
        return
    _beep_async(1000, 80)


def play_end_sound():
    """Звук окончания записи."""
    if not SOUND_ENABLED:
        return
    if _sound_end_file and _play_audio_file(str(_sound_end_file)):
        return
    _beep_async(500, 80)


def play_ready_sound():
    """Звук готовности (после распознавания или загрузки модели)."""
    if not SOUND_ENABLED:
        return
    if _sound_ready_file and _play_audio_file(str(_sound_ready_file)):
        return
    _beep_async(750, 100)


# ---------- Log & internal buffer ----------
_internal_buffer = deque(maxlen=50)


def log_transcription(text):
    _internal_buffer.append(text)

    if not LOG_ENABLED:
        return

    try:
        ts = _time.strftime("%Y-%m-%d %H:%M:%S")

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {text}\n")

    except Exception as e:
        print(f"[log] write failed: {e}")


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
        print(f"[tray] opened log: {LOG_FILE}")
    except Exception as e:
        print(f"[tray] open log failed: {e}")


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

        print(f"[tray] copied last to clipboard: {last!r}")

        if icon_:
            icon_.notify(
                f"Скопировано: {last[:60]}",
                "Whisper PTT"
            )

    except Exception as e:
        print(f"[tray] copy failed: {e}")


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

    root.title("Whisper PTT — история расшифровок")
    root.geometry("700x450")

    top = tk.Frame(root)
    top.pack(fill="x", padx=8, pady=6)

    text_widget = None

    def copy_selected():
        try:
            sel = text_widget.get("sel.first", "sel.last")
        except tk.TclError:
            sel = ""

        if sel:
            root.clipboard_clear()
            root.clipboard_append(sel)
            print(f"[ui] copied selection: {sel[:60]!r}")

    def copy_all():
        all_text = text_widget.get("1.0", "end").strip()

        root.clipboard_clear()
        root.clipboard_append(all_text)

        print(f"[ui] copied all ({len(all_text)} chars)")

    def clear_view():
        text_widget.delete("1.0", "end")

    def refresh_view():
        text_widget.delete("1.0", "end")

        for i, t in enumerate(_internal_buffer, 1):
            text_widget.insert("end", f"{i:>3}. {t}\n")

    tk.Button(
        top,
        text="Копировать выделенное",
        command=copy_selected
    ).pack(side="left", padx=2)

    tk.Button(
        top,
        text="Копировать всё",
        command=copy_all
    ).pack(side="left", padx=2)

    tk.Button(
        top,
        text="Очистить вид",
        command=clear_view
    ).pack(side="left", padx=2)

    tk.Button(
        top,
        text="Обновить",
        command=refresh_view
    ).pack(side="left", padx=2)

    tk.Label(
        top,
        text=f"(буфер: до {_internal_buffer.maxlen} шт., в файле: {LOG_FILE.name})"
    ).pack(side="right")

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=8, pady=6)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    text_widget = tk.Text(
        frame,
        wrap="word",
        yscrollcommand=scrollbar.set,
        font=("Consolas", 11)
    )
    text_widget.pack(side="left", fill="both", expand=True)

    scrollbar.config(command=text_widget.yview)

    refresh_view()
    text_widget.see("end")

    def on_close():
        global _history_window
        _history_window = None
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
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

# Anti-spam: игнорируем авто-повтор клавиши при удержании
_hotkey_held = False

COLOR_IDLE = (60, 170, 90)
COLOR_REC  = (220, 60, 60)
COLOR_LOAD = (180, 160, 60)
COLOR_ERR  = (120, 120, 120)


def make_image(color):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.ellipse((4, 4, 60, 60), fill=color)

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
def _find_local_model():
    """Ищет локальную модель. Интернет здесь не используется."""
    for cand in [
        Path(MODEL_NAME),
        APP_DIR / MODEL_NAME,
        APP_DIR / "models" / MODEL_NAME,
        APP_DIR / "models",
        Path("C:/models") / MODEL_NAME,
    ]:
        if cand.is_dir() and (cand / "model.bin").exists():
            return cand.resolve()

    return None


def load_model():
    global model

    try:
        set_tray_status("загрузка модели…")
        set_color(COLOR_LOAD)

        local_dir = _find_local_model()

        if local_dir is None:
            msg = (
                f"Локальная модель '{MODEL_NAME}' не найдена. "
                f"Положите модель в папку models/ рядом со скриптом. "
                f"Скачивание из интернета отключено."
            )
            print(f"[whisper] {msg}")
            set_color(COLOR_ERR)
            set_tray_status("ошибка: модель не найдена")
            if icon:
                try:
                    icon.notify(msg, "Whisper PTT")
                except Exception:
                    pass
            return

        print(f"[whisper] loading local model: {local_dir}")
        print("[whisper] no internet download required")

        if BACKEND == "faster":
            from faster_whisper import WhisperModel

            model = WhisperModel(
                str(local_dir),
                device=DEVICE,
                compute_type=COMPUTE_TYPE
            )

        elif BACKEND == "openai":
            raise ValueError(
                "backend=openai отключён в offline-режиме "
                "(может скачивать модель из интернета). "
                "Используйте backend=faster с локальной CTranslate2-моделью."
            )

        else:
            raise ValueError(f"Unknown backend: {BACKEND}")

        model_ready.set()
        set_color(COLOR_IDLE)
        set_tray_status(None)

        print(f"[whisper] READY ({BACKEND}, {DEVICE}, {COMPUTE_TYPE})")

        play_ready_sound()

    except Exception as e:
        print(f"[whisper] FAILED: {e}")

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
def _audio_cb(indata, frames, time_info, status):
    if recording:
        audio_chunks.append(indata.copy())
    else:
        _preroll_buffer.append(indata.copy())


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
            f"[audio] failed to start audio stream: {e}"
        )


def start_recording():
    global recording, audio_chunks

    if not model_ready.is_set():
        print("[rec] model not ready")
        return

    with stream_lock:
        if recording:
            return

        # Захватываем pre-roll буфер, чтобы не потерять начало фразы.
        audio_chunks = list(_preroll_buffer)
        _preroll_buffer.clear()

        recording = True

    set_color(COLOR_REC)

    if icon:
        icon.title = (
            f"Whisper PTT — REC "
            f"({HOTKEY_STR}, {HOTKEY_MODE})"
        )

    print("[rec] started")
    play_start_sound()


def stop_recording():
    global recording

    with stream_lock:
        if not recording:
            return

        recording = False
        chunks = list(audio_chunks)

    set_color(COLOR_IDLE)

    if icon:
        icon.title = (
            f"Whisper PTT — transcribing… "
            f"({HOTKEY_STR}, {HOTKEY_MODE})"
        )

    print(
        "[rec] stopped, queued for transcription"
    )

    play_end_sound()

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

    # Отсекаем слишком короткие клики (< 0.35 сек).
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

    # Фильтр галлюцинаций Whisper — фразы которые модель придумывает
    # из тишины или шума, особенно на русском языке.
    _HALLUCINATIONS = [
        "субтитры сделал dimatorzok",
        "субтитры сделал",
        "субтитры сделал dimatorzok.",
        "спасибо за просмотр",
        "подписывайтесь на канал",
        "приятного просмотра",
        "вы можете помочь развитию канала",
        "я в вк",
        "музыка",
    ]
    _text_lower = text.lower()
    for _hall in _HALLUCINATIONS:
        if _hall in _text_lower:
            print(f"[rec] hallucination filtered: {_hall!r}")
            text = text.replace(_hall, "")
            # Также убираем с большой буквы
            text = text.replace(_hall.capitalize(), "")
    text = text.strip()

    # Если после фильтрации ничего не осталось — пропускаем
    if not text:
        print("[rec] empty after hallucination filter")
        if icon:
            icon.title = (
                f"Whisper PTT — ready "
                f"({HOTKEY_STR}, {HOTKEY_MODE})"
            )
        return

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
        print("[rec] empty result")

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

    _send_key(VK_LCONTROL, up=False)
    _send_key(VK_V, up=False)

    _time.sleep(0.01)

    _send_key(VK_V, up=True)
    _send_key(VK_LCONTROL, up=True)

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
    global _hotkey_held

    if not _is_hotkey(key):
        return

    # Игнорируем авто-повтор при удержании клавиши
    if _hotkey_held:
        return

    _hotkey_held = True

    if HOTKEY_MODE == "toggle":
        # Нажал F9:
        #   не записываем -> начинаем
        #   записываем -> останавливаем
        if recording:
            stop_recording()
        else:
            start_recording()

        return

    # HOLD
    if recording:
        return

    start_recording()


def on_release(key):
    global _hotkey_held

    if not _is_hotkey(key):
        return

    _hotkey_held = False

    # В toggle отпускание кнопки ничего не делает.
    if HOTKEY_MODE == "toggle":
        return

    # HOLD
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

    # Закрываем все MCI устройства
    try:
        _winmm.mciSendStringW("close all", None, 0, None)
    except Exception:
        pass

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
            f"Whisper PTT — загрузка модели… "
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
                f"Устройство: {DEVICE} / {COMPUTE_TYPE}",
                None,
                enabled=False
            ),

            MenuItem(
                f"Метод вставки: {PASTE_METHOD}",
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

    # Загружаем звуки ДО модели, чтобы сигнал готовности
    # гарантированно проигрался.
    load_sound()

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
