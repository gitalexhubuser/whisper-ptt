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

# ============ CTranslate2 shim ============
if "pkg_resources" not in sys.modules:
    class _PRShim:
        @staticmethod
        def resource_filename(mod, rel=""):
            import importlib.util
            spec = importlib.util.find_spec(mod)
            p = os.path.dirname(spec.origin) if (spec and spec.origin) else ""
            return os.path.join(p, rel)
    sys.modules["pkg_resources"] = _PRShim()

# ============ WinAPI SendInput ============
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


def _send_key(vk, up=False):
    inp = _INPUT()
    inp.type = INPUT_KEYBOARD
    inp.u.ki.wVk = vk
    inp.u.ki.wScan = ctypes.windll.user32.MapVirtualKeyW(vk, 0)
    inp.u.ki.dwFlags = KEYEVENTF_KEYUP if up else 0
    inp.u.ki.time = 0
    inp.u.ki.dwExtraInfo = None
    return ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def _send_unicode_batch(text):
    inputs = []
    for ch in text:
        code = ord(ch)
        if code > 0xFFFF:
            code -= 0x10000
            high = 0xD800 + (code >> 10)
            low  = 0xDC00 + (code & 0x3FF)
            for scan in (high, low):
                inp_d = _INPUT()
                inp_d.type = INPUT_KEYBOARD
                inp_d.u.ki.wVk = 0
                inp_d.u.ki.wScan = scan
                inp_d.u.ki.dwFlags = KEYEVENTF_UNICODE
                inp_d.u.ki.time = 0
                inp_d.u.ki.dwExtraInfo = None
                inputs.append(inp_d)

                inp_u = _INPUT()
                inp_u.type = INPUT_KEYBOARD
                inp_u.u.ki.wVk = 0
                inp_u.u.ki.wScan = scan
                inp_u.u.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
                inp_u.u.ki.time = 0
                inp_u.u.ki.dwExtraInfo = None
                inputs.append(inp_u)
        else:
            inp_d = _INPUT()
            inp_d.type = INPUT_KEYBOARD
            inp_d.u.ki.wVk = 0
            inp_d.u.ki.wScan = code
            inp_d.u.ki.dwFlags = KEYEVENTF_UNICODE
            inp_d.u.ki.time = 0
            inp_d.u.ki.dwExtraInfo = None
            inputs.append(inp_d)

            inp_u = _INPUT()
            inp_u.type = INPUT_KEYBOARD
            inp_u.u.ki.wVk = 0
            inp_u.u.ki.wScan = code
            inp_u.u.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
            inp_u.u.ki.time = 0
            inp_u.u.ki.dwExtraInfo = None
            inputs.append(inp_u)

    if not inputs:
        return 0

    c_inputs = (_INPUT * len(inputs))(*inputs)
    return ctypes.windll.user32.SendInput(len(inputs), c_inputs, ctypes.sizeof(_INPUT))


def _release_all_modifiers():
    for vk in (VK_LCONTROL, VK_LSHIFT, VK_LALT, VK_LWIN):
        _send_key(vk, up=True)


# =========================================

import configparser
import queue
import threading
from collections import deque
from pathlib import Path

import keyboard  # Используем системный хук Windows
import numpy as np
import sounddevice as sd
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
key = f8

[audio]
sample_rate = 16000

[text]
add_trailing_space = true

[paste]
method = unicode
char_delay = 0.0

[log]
enabled = true
file = transcriptions.log

[sound]
enabled = true
start_file = start.mp3
end_file = end.mp3
ready_file = ready.mp3
volume = 0.7
"""


def ensure_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(DEFAULT_CONFIG, encoding="utf-8")

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(CONFIG_PATH, encoding="utf-8")
    return cfg


cfg = ensure_config()

BACKEND      = cfg.get("whisper", "backend", fallback="faster").strip().lower()
MODEL_NAME   = cfg.get("whisper", "model", fallback="large-v3-turbo").strip()
DEVICE       = cfg.get("whisper", "device", fallback="cuda").strip()
COMPUTE_TYPE = cfg.get("whisper", "compute_type", fallback="int8_float32").strip()
LANGUAGE     = cfg.get("whisper", "language", fallback="ru").strip() or None
BEAM_SIZE    = cfg.getint("whisper", "beam_size", fallback=1)

HOTKEY_STR   = cfg.get("hotkey", "key", fallback="f8").strip().lower()

SAMPLE_RATE  = cfg.getint("audio", "sample_rate", fallback=16000)
ADD_SPACE    = cfg.getboolean("text", "add_trailing_space", fallback=True)

PASTE_METHOD = cfg.get("paste", "method", fallback="unicode").strip().lower()
CHAR_DELAY   = cfg.getfloat("paste", "char_delay", fallback=0.0)

LOG_ENABLED  = cfg.getboolean("log", "enabled", fallback=True)
LOG_FILE     = APP_DIR / cfg.get("log", "file", fallback="transcriptions.log").strip()

SOUND_ENABLED = cfg.getboolean("sound", "enabled", fallback=True)
SOUND_VOLUME  = cfg.getfloat("sound", "volume", fallback=0.7)

_ready_fallback = cfg.get("sound", "file", fallback="ready.mp3").strip()
SOUND_START_FILE = cfg.get("sound", "start_file", fallback="start.mp3").strip()
SOUND_END_FILE   = cfg.get("sound", "end_file", fallback="end.mp3").strip()
SOUND_READY_FILE = cfg.get("sound", "ready_file", fallback=_ready_fallback).strip()

# ---------- Sound (winmm.dll) ----------
_winmm = ctypes.windll.winmm
_winmm.mciSendStringW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_void_p]
_winmm.mciSendStringW.restype = ctypes.c_uint

_mci_lock = threading.Lock()
_mci_counter = 0

_sound_start_file = None
_sound_end_file = None
_sound_ready_file = None


def _play_audio_file(filepath):
    global _mci_counter
    filepath = os.path.abspath(str(filepath))

    with _mci_lock:
        _mci_counter += 1
        alias = f"wpsnd_{_mci_counter}"
        try:
            cmd = f'open "{filepath}" alias {alias}'
            ret = _winmm.mciSendStringW(cmd, None, 0, None)
            if ret != 0:
                cmd = f'open "{filepath}" type mpegvideo alias {alias}'
                ret = _winmm.mciSendStringW(cmd, None, 0, None)
                if ret != 0:
                    return False

            vol = max(0, min(1000, int(SOUND_VOLUME * 1000)))
            _winmm.mciSendStringW(f'setaudio {alias} volume to {vol}', None, 0, None)
            _winmm.mciSendStringW(f'play {alias}', None, 0, None)
        except Exception:
            return False

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
    def _b():
        try:
            import winsound
            winsound.Beep(freq, duration)
        except Exception:
            pass
    threading.Thread(target=_b, daemon=True).start()


def load_sound():
    global _sound_start_file, _sound_end_file, _sound_ready_file
    if not SOUND_ENABLED:
        return

    for attr, filename in [
        ("_sound_start_file", SOUND_START_FILE),
        ("_sound_end_file",   SOUND_END_FILE),
        ("_sound_ready_file", SOUND_READY_FILE),
    ]:
        path = APP_DIR / filename
        if path.exists():
            globals()[attr] = path


def play_start_sound():
    if not SOUND_ENABLED:
        return
    if _sound_start_file and _play_audio_file(str(_sound_start_file)):
        return
    _beep_async(1000, 80)


def play_end_sound():
    if not SOUND_ENABLED:
        return
    if _sound_end_file and _play_audio_file(str(_sound_end_file)):
        return
    _beep_async(500, 80)


def play_ready_sound():
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
        return
    try:
        os.startfile(str(LOG_FILE))
    except Exception as e:
        print(f"[tray] open log failed: {e}")


def copy_last_to_clipboard(icon_, item):
    if not _internal_buffer:
        return
    try:
        import pyperclip
        pyperclip.copy(_internal_buffer[-1])
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

    def copy_all():
        all_text = text_widget.get("1.0", "end").strip()
        root.clipboard_clear()
        root.clipboard_append(all_text)

    def clear_view():
        text_widget.delete("1.0", "end")

    def refresh_view():
        text_widget.delete("1.0", "end")
        for i, t in enumerate(_internal_buffer, 1):
            text_widget.insert("end", f"{i:>3}. {t}\n")

    tk.Button(top, text="Копировать всё", command=copy_all).pack(side="left", padx=2)
    tk.Button(top, text="Очистить вид", command=clear_view).pack(side="left", padx=2)
    tk.Button(top, text="Обновить", command=refresh_view).pack(side="left", padx=2)

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=8, pady=6)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    text_widget = tk.Text(frame, wrap="word", yscrollcommand=scrollbar.set, font=("Consolas", 11))
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
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill=color)
    d.rounded_rectangle((26, 14, 38, 40), radius=6, fill=(255, 255, 255))
    d.arc((18, 26, 46, 50), start=0, end=180, fill=(255, 255, 255), width=3)
    d.line((32, 46, 32, 54), fill=(255, 255, 255), width=3)
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
                icon.title = f"Whisper PTT — {status_text} ({HOTKEY_STR}, toggle)"
            else:
                icon.title = f"Whisper PTT — ready ({HOTKEY_STR}, toggle)"
        except Exception:
            pass


# ---------- Model ----------
def _find_local_model():
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
            msg = f"Локальная модель '{MODEL_NAME}' не найдена."
            print(f"[whisper] {msg}")
            set_color(COLOR_ERR)
            set_tray_status("ошибка: модель не найдена")
            return

        print(f"[whisper] loading local model: {local_dir}")
        from faster_whisper import WhisperModel
        model = WhisperModel(str(local_dir), device=DEVICE, compute_type=COMPUTE_TYPE)

        model_ready.set()
        set_color(COLOR_IDLE)
        set_tray_status(None)
        print(f"[whisper] READY ({BACKEND}, {DEVICE}, {COMPUTE_TYPE})")
        play_ready_sound()

    except Exception as e:
        print(f"[whisper] FAILED: {e}")
        set_color(COLOR_ERR)


# ---------- Audio Stream ----------
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
        print("[audio] persistent stream active")
    except Exception as e:
        print(f"[audio] stream error: {e}")


def toggle_recording():
    """Атомарный обработчик включения/выключения записи."""
    global recording, audio_chunks

    if not model_ready.is_set():
        print("[rec] model not ready yet")
        return

    with stream_lock:
        if not recording:
            # ВКЛЮЧАЕМ
            audio_chunks = list(_preroll_buffer)
            _preroll_buffer.clear()
            recording = True
            set_color(COLOR_REC)
            set_tray_status("REC")
            print(f"[rec] started ({HOTKEY_STR} toggle ON)")
            threading.Thread(target=play_start_sound, daemon=True).start()
        else:
            # ВЫКЛЮЧАЕМ
            recording = False
            chunks = list(audio_chunks)
            set_color(COLOR_IDLE)
            set_tray_status("transcribing…")
            print(f"[rec] stopped ({HOTKEY_STR} toggle OFF), transcribing...")
            threading.Thread(target=play_end_sound, daemon=True).start()
            _audio_queue.put(chunks)


def _worker_loop():
    while True:
        chunks = _audio_queue.get()
        try:
            process_audio(chunks)
        except Exception as e:
            print(f"[worker] process error: {e}")
        finally:
            _audio_queue.task_done()


def process_audio(chunks):
    if not chunks:
        return

    try:
        audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
    except Exception as e:
        print(f"[rec] concat error: {e}")
        return

    if len(audio) < SAMPLE_RATE * 0.15:
        print("[rec] too short, skip")
        set_tray_status(None)
        return

    t0 = _time.time()
    try:
        with model_lock:
            segments, _ = model.transcribe(
                audio,
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                temperature=0.0,
                without_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
                condition_on_previous_text=False,
            )
            text = "".join(s.text for s in segments)
    except Exception as e:
        print(f"[rec] transcribe error: {e}")
        set_tray_status(None)
        return

    text = text.strip()

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
            text = text.replace(_hall, "").replace(_hall.capitalize(), "")
    text = text.strip()

    if not text:
        set_tray_status(None)
        return

    dur = len(audio) / SAMPLE_RATE
    dt = _time.time() - t0
    print(f"[rec] {dur:.1f}s audio -> {dt:.2f}s -> {text!r}")

    log_transcription(text)
    out = (text + " ") if ADD_SPACE else text

    if PASTE_METHOD == "unicode":
        type_text_to_active_window(out)
    else:
        paste_via_clipboard(out)

    play_ready_sound()
    set_tray_status(None)


def type_text_to_active_window(text):
    _release_all_modifiers()
    _time.sleep(0.01)
    _send_unicode_batch(text)


def paste_via_clipboard(text):
    try:
        import pyperclip
        pyperclip.copy(text)
    except Exception:
        return

    _release_all_modifiers()
    _time.sleep(0.02)
    _send_key(VK_LCONTROL, up=False)
    _send_key(VK_V, up=False)
    _time.sleep(0.01)
    _send_key(VK_V, up=True)
    _send_key(VK_LCONTROL, up=True)


# ---------- Listener via keyboard library ----------
def setup_hotkey():
    """
    Использует нативный хук WH_KEYBOARD_LL библиотеки keyboard.
    Срабатывает всегда с 1-го нажатия в любых окнах.
    """
    print(f"[hotkey] registering '{HOTKEY_STR}' via native hook...")

    # on_press_key ловит только момент вдавливания клавиши (down), игнорируя отпускание
    keyboard.on_press_key(HOTKEY_STR, lambda e: toggle_recording())


# ---------- Tray ----------
def on_quit(icon_, item):
    global stream
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
        keyboard.unhook_all()
    except Exception:
        pass

    if icon_:
        icon_.stop()
    os._exit(0)


def main():
    global icon

    icon = Icon(
        "whisper-ptt",
        make_image(COLOR_LOAD),
        title=f"Whisper PTT — загрузка… ({HOTKEY_STR}, toggle)",
        menu=Menu(
            MenuItem(f"Хоткей: {HOTKEY_STR} (toggle)", None, enabled=False),
            MenuItem(f"Модель: {MODEL_NAME}", None, enabled=False),
            MenuItem(f"Устройство: {DEVICE} / {COMPUTE_TYPE}", None, enabled=False),
            Menu.SEPARATOR,
            MenuItem("Открыть историю расшифровок", open_history_window, default=True),
            MenuItem("Открыть лог-файл", open_log_file),
            MenuItem("Скопировать последнюю в буфер", copy_last_to_clipboard),
            Menu.SEPARATOR,
            MenuItem("Выход", on_quit),
        ),
    )

    init_audio_stream()
    threading.Thread(target=_worker_loop, daemon=True).start()
    load_sound()
    threading.Thread(target=load_model, daemon=True).start()

    # Запускаем хук клавиатуры
    setup_hotkey()

    icon.run()


if __name__ == "__main__":
    main()
    