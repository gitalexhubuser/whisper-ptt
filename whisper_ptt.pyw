# ============ CUDA DLL PATCH ============
import os
import sys

if sys.platform == "win32":
    try:
        import site
        for sp in site.getsitepackages():
            for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
                p = os.path.join(sp, sub.replace("/", os.sep))
                if os.path.isdir(p):
                    os.add_dll_directory(p)
                    os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
                    print(f"[cuda] DLL dir: {p}")
    except Exception as e:
        print(f"[cuda] patch failed: {e}")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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
        ("wScan",       wintypes.WORD),
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
        ("u",    _INPUT_UNION),
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

[audio]
sample_rate = 16000

[text]
add_trailing_space = true

[paste]
method = unicode
char_delay = 0.001

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
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(CONFIG_PATH, encoding="utf-8")
    return cfg


cfg = ensure_config()

BACKEND      = cfg.get("whisper", "backend",      fallback="faster").strip().lower()
MODEL_NAME   = cfg.get("whisper", "model",        fallback="large-v3-turbo").strip()
DEVICE       = cfg.get("whisper", "device",       fallback="cuda").strip()
COMPUTE_TYPE = cfg.get("whisper", "compute_type", fallback="int8_float32").strip()
LANGUAGE     = cfg.get("whisper", "language",     fallback="ru").strip() or None
BEAM_SIZE    = cfg.getint("whisper", "beam_size", fallback=1)

HOTKEY_STR   = cfg.get("hotkey", "key", fallback="f9").strip().lower()
SAMPLE_RATE  = cfg.getint("audio",  "sample_rate", fallback=16000)
ADD_SPACE    = cfg.getboolean("text", "add_trailing_space", fallback=True)

PASTE_METHOD = cfg.get("paste", "method", fallback="unicode").strip().lower()
CHAR_DELAY   = cfg.getfloat("paste", "char_delay", fallback=0.001)

LOG_ENABLED  = cfg.getboolean("log", "enabled", fallback=True)
LOG_FILE     = APP_DIR / cfg.get("log", "file", fallback="transcriptions.log").strip()

SOUND_ENABLED = cfg.getboolean("sound", "enabled", fallback=True)
SOUND_FILE    = APP_DIR / cfg.get("sound", "file", fallback="ready.mp3").strip()
SOUND_VOLUME  = cfg.getfloat("sound", "volume", fallback=0.7)


# ---------- Hotkey resolve (одна конкретная клавиша) ----------
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
print(f"[hotkey] using: {HOTKEY_STR} ({HOTKEY})")


# ---------- Sound ----------
_sound_obj = None
_sound_ready = False


def load_sound():
    global _sound_obj, _sound_ready
    if not SOUND_ENABLED:
        print("[sound] disabled")
        return
    if not SOUND_FILE.exists():
        print(f"[sound] file not found: {SOUND_FILE} — звук отключён")
        return
    try:
        import pygame
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        _sound_obj = pygame.mixer.Sound(str(SOUND_FILE))
        _sound_obj.set_volume(SOUND_VOLUME)
        _sound_ready = True
        print(f"[sound] loaded: {SOUND_FILE.name}")
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
        ts = _time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {text}\n")
    except Exception as e:
        print(f"[log] write failed: {e}")


def open_log_file(icon_, item):
    if not LOG_FILE.exists():
        if icon_:
            try:
                icon_.notify(f"Лог ещё пуст: {LOG_FILE.name}", "Whisper PTT")
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
            icon_.notify("Буфер расшифровок пуст", "Whisper PTT")
        return
    last = _internal_buffer[-1]
    try:
        import pyperclip
        pyperclip.copy(last)
        print(f"[tray] copied last to clipboard: {last!r}")
        if icon_:
            icon_.notify(f"Скопировано: {last[:60]}", "Whisper PTT")
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

    tk.Button(top, text="Копировать выделенное", command=copy_selected).pack(side="left", padx=2)
    tk.Button(top, text="Копировать всё", command=copy_all).pack(side="left", padx=2)
    tk.Button(top, text="Очистить вид", command=clear_view).pack(side="left", padx=2)
    tk.Button(top, text="Обновить", command=refresh_view).pack(side="left", padx=2)
    tk.Label(top, text=f"(буфер: до {_internal_buffer.maxlen} шт., в файле: {LOG_FILE.name})").pack(side="right")

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=8, pady=6)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    text_widget = tk.Text(frame, wrap="word", yscrollcommand=scrollbar.set,
                          font=("Consolas", 11))
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


# ---------- State ----------
model = None
model_lock = threading.Lock()
model_ready = threading.Event()

recording = False
audio_chunks = []
stream = None
stream_lock = threading.Lock()

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


# ---------- Model ----------
def load_model():
    global model
    try:
        if BACKEND == "faster":
            from faster_whisper import WhisperModel
            print(f"[whisper] loading {MODEL_NAME} on {DEVICE} ({COMPUTE_TYPE})…")
            model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
        elif BACKEND == "openai":
            import whisper
            model = whisper.load_model(MODEL_NAME, device=DEVICE)
        else:
            raise ValueError(f"Unknown backend: {BACKEND}")

        model_ready.set()
        set_color(COLOR_IDLE)
        print(f"[whisper] READY ({BACKEND}, {DEVICE}, {COMPUTE_TYPE})")
        if icon:
            icon.title = f"Whisper PTT — ready ({HOTKEY_STR})"

    except Exception as e:
        print(f"[whisper] FAILED: {e}")
        set_color(COLOR_ERR)
        if icon:
            try:
                icon.notify(f"Ошибка загрузки модели: {e}", "Whisper PTT")
            except Exception:
                pass


# ---------- Recording ----------
def _audio_cb(indata, frames, time_info, status):
    if recording:
        audio_chunks.append(indata.copy())


def start_recording():
    global recording, stream, audio_chunks
    if not model_ready.is_set():
        print("[rec] model not ready")
        return
    with stream_lock:
        if recording:
            return
        audio_chunks = []
        recording = True
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            callback=_audio_cb,
        )
        stream.start()
    set_color(COLOR_REC)
    if icon:
        icon.title = f"Whisper PTT — REC ({HOTKEY_STR})"
    print("[rec] started")


def stop_recording():
    global recording, stream
    with stream_lock:
        if not recording:
            return
        recording = False
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            stream = None
        chunks = list(audio_chunks)
    set_color(COLOR_IDLE)
    if icon:
        icon.title = f"Whisper PTT — transcribing… ({HOTKEY_STR})"
    print("[rec] stopped")
    threading.Thread(target=process_audio, args=(chunks,), daemon=True).start()


def process_audio(chunks):
    if not chunks:
        return
    try:
        audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
    except Exception as e:
        print(f"[rec] concat error: {e}")
        return

    if len(audio) < SAMPLE_RATE * 0.4:
        print("[rec] too short, skip")
        if icon:
            icon.title = f"Whisper PTT — ready ({HOTKEY_STR})"
        return

    t0 = _time.time()
    try:
        with model_lock:
            if BACKEND == "faster":
                segments, _ = model.transcribe(
                    audio,
                    language=LANGUAGE,
                    beam_size=BEAM_SIZE,
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 700,
                        "speech_pad_ms": 400,
                    },
                    condition_on_previous_text=False,
                )
                text = "".join(s.text for s in segments)
            else:
                result = model.transcribe(
                    audio,
                    language=LANGUAGE,
                    fp16=(DEVICE == "cuda"),
                    condition_on_previous_text=False,
                )
                text = result.get("text", "")
    except Exception as e:
        print(f"[rec] transcribe error: {e}")
        if icon:
            icon.title = f"Whisper PTT — ready ({HOTKEY_STR})"
        return

    text = text.strip()
    dur = len(audio) / SAMPLE_RATE
    print(f"[rec] {dur:.1f}s audio -> {_time.time()-t0:.2f}s -> {text!r}")

    if text:
        log_transcription(text)

        out = text + " " if ADD_SPACE else text
        if PASTE_METHOD == "unicode":
            type_text_to_active_window(out)
        else:
            paste_via_clipboard(out)

        play_ready_sound()
    else:
        print("[rec] empty result")

    if icon:
        icon.title = f"Whisper PTT — ready ({HOTKEY_STR})"


def type_text_to_active_window(text):
    _release_all_modifiers()
    _time.sleep(0.05)
    t0 = _time.time()
    try:
        _type_text(text, char_delay=CHAR_DELAY)
        dt = _time.time() - t0
        print(f"[type] sent {len(text)} chars in {dt:.2f}s")
    except Exception as e:
        print(f"[type] failed: {e}")


def paste_via_clipboard(text):
    try:
        import pyperclip
        pyperclip.copy(text)
    except Exception as e:
        print(f"[paste] clipboard error: {e}")
        return

    _time.sleep(0.1)
    _release_all_modifiers()
    _time.sleep(0.25)

    n1 = _send_key(VK_LCONTROL, up=False); _time.sleep(0.03)
    n2 = _send_key(VK_V,        up=False); _time.sleep(0.03)
    n3 = _send_key(VK_V,        up=True);  _time.sleep(0.03)
    n4 = _send_key(VK_LCONTROL, up=True)
    print(f"[paste] SendInput total events: {n1+n2+n3+n4}/4")


# ---------- Listener ----------
def _is_hotkey(key):
    """Сравнение с учётом KeyCode (char) и Key (enum)."""
    if key == HOTKEY:
        return True
    if isinstance(HOTKEY, keyboard.KeyCode) and isinstance(key, keyboard.KeyCode):
        return HOTKEY.char == key.char
    return False


def on_press(key):
    if not _is_hotkey(key):
        return
    if recording:
        return  # автоповтор F9 — игнор
    start_recording()


def on_release(key):
    if not _is_hotkey(key):
        return
    if not recording:
        return
    stop_recording()


def run_listener():
    with keyboard.Listener(on_press=on_press, on_release=on_release) as l:
        l.join()


# ---------- Tray ----------
def on_quit(icon_, item):
    try:
        icon_.stop()
    finally:
        os._exit(0)


def main():
    global icon
    icon = Icon(
        "whisper-ptt",
        make_image(COLOR_LOAD),
        title=f"Whisper PTT — loading… ({HOTKEY_STR})",
        menu=Menu(
            MenuItem(f"Хоткей: {HOTKEY_STR}", None, enabled=False),
            MenuItem(f"Модель: {MODEL_NAME}", None, enabled=False),
            MenuItem(f"Устройство: {DEVICE} / {COMPUTE_TYPE}", None, enabled=False),
            MenuItem(f"Метод вставки: {PASTE_METHOD}", None, enabled=False),
            Menu.SEPARATOR,
            MenuItem("Открыть историю расшифровок", open_history_window, default=True),
            MenuItem("Открыть лог-файл", open_log_file),
            MenuItem("Скопировать последнюю в буфер", copy_last_to_clipboard),
            Menu.SEPARATOR,
            MenuItem("Выход", on_quit),
        ),
    )

    threading.Thread(target=load_sound, daemon=True).start()
    threading.Thread(target=load_model, daemon=True).start()
    threading.Thread(target=run_listener, daemon=True).start()

    icon.run()


if __name__ == "__main__":
    main()