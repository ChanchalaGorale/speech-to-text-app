"""Push-to-talk dictation: hold Ctrl+Space, speak, release — text types at the cursor.

Default PTT is Ctrl+Space (no streaming while held — transcribe once on release, so Ctrl
is not held during typing). Use DICTATE_PTT for other chords (e.g. ctrl_shift_space).
"""

import os
import platform
import subprocess
import sys
import threading
import time

import numpy as np
import pyautogui
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard as kb
from pynput.keyboard import Controller as TypeController

# Default: Ctrl+Space (two keys). Plain Space alone is not used (repeat clears the buffer).
_DEFAULT_PTT = "ctrl_space"

_TRIPLE_PRESETS = {
    "ctrl_shift_space": ("ctrl", "shift", "space"),
    "cmd_shift_space": ("cmd", "shift", "space"),
    "ctrl_alt_space": ("ctrl", "alt", "space"),
}

# On MacBooks, the top row often sends *media* keys unless “Use F1, F2, … as standard function keys”
# is on. Same physical key: F7 ↔ previous track, F8 ↔ play/pause, F9 ↔ next, etc.
_MAC_FN_EQUIVALENTS = {
    kb.Key.f7: (kb.Key.media_previous,),
    kb.Key.f8: (kb.Key.media_play_pause,),
    kb.Key.f9: (kb.Key.media_next,),
    kb.Key.f10: (kb.Key.media_volume_mute,),
    kb.Key.f11: (kb.Key.media_volume_down,),
    kb.Key.f12: (kb.Key.media_volume_up,),
}

# Loaded on first recording (so keyboard listener can start before a long model download)
_model = None
_model_lock = threading.Lock()

fs = 16000
recording = []
is_recording = False
last_emit_sample = 0
_state_lock = threading.Lock()

# Keys currently held (combo / triple / debugging)
_pressed: set = set()
_warned_untrusted_inject = False

# Typing into the frontmost app (Electron / Cursor often works better than clipboard paste)
_typer = TypeController()
pyautogui.PAUSE = 0.02

# Streaming: wake interval and slice sizes (seconds of audio)
STREAM_WAKE_S = 0.55
STREAM_INTERVAL_S = 1.05
STREAM_MIN_NEW_S = 0.45
STREAM_MAX_CHUNK_S = 2.8

# When macOS is not trusted: auto Cmd+V at most this often for streaming partials (ms).
_UNTRUSTED_PASTE_INTERVAL_S = float(os.environ.get("DICTATE_UNTRUSTED_PASTE_MS", "500")) / 1000.0
_last_untrusted_paste_ts = 0.0
_untrusted_paste_lock = threading.Lock()

# macOS AX: only inject into real text controls (requires Accessibility trust).
_warned_focus_ax = False
_last_focus_reject_reason = None
_ALLOWED_TEXT_ROLES = frozenset(
    {
        "AXTextField",
        "AXTextArea",
        "AXComboBox",
        "AXSearchField",
        "AXDateField",
        "AXTimeField",
        # Subroles (from kAXSubroleAttribute) appear in the same tag list
        "AXSecureTextField",
    }
)


def _ensure_model():
    global _model
    with _model_lock:
        if _model is None:
            print("Loading Whisper model (first use)…")
            _model = WhisperModel("base", compute_type="int8")
        return _model


def _parse_ptt():
    """Returns (mode, key|None, triple_tokens|None). Modes: combo | triple | key."""
    raw = os.environ.get("DICTATE_PTT", _DEFAULT_PTT).strip().lower()
    if raw in ("ctrl_space", "ctrl+space", "ctrl-space"):
        return "combo", None, None
    if raw in _TRIPLE_PRESETS:
        return "triple", None, _TRIPLE_PRESETS[raw]
    key = getattr(kb.Key, raw, None)
    if key is None:
        print(
            f"Unknown DICTATE_PTT={raw!r} — using {_DEFAULT_PTT}. "
            "Try: ctrl_space, ctrl_shift_space, f7, pause"
        )
        fd = _DEFAULT_PTT.strip().lower()
        if fd in ("ctrl_space", "ctrl+space", "ctrl-space"):
            return "combo", None, None
        if fd in _TRIPLE_PRESETS:
            return "triple", None, _TRIPLE_PRESETS[fd]
        return "combo", None, None
    return "key", key, None


_PTT_MODE, _PTT_KEY, _PTT_TRIPLE = _parse_ptt()


def _ptt_key_variants(primary: kb.Key) -> frozenset:
    """KeyCode variants for PTT (listeners get KeyCode, e.g. <98>, not Key.f7 — compare .value).

    macOS F-row: same physical key may send F7 (vk 98) or media_previous (vk 18).
    """
    s = {primary.value}
    if platform.system() != "Darwin":
        return frozenset(s)
    for fn_key, alts in _MAC_FN_EQUIVALENTS.items():
        if primary == fn_key:
            s.update(k.value for k in alts)
            return frozenset(s)
        if primary in alts:
            s.add(fn_key.value)
            return frozenset(s)
    return frozenset(s)


# Match by vk only: listener sends KeyCode; media keys have _is_media so == Key.f7.value can fail.
_PTT_VKS = (
    frozenset(
        _kc.vk
        for _kc in _ptt_key_variants(_PTT_KEY)
        if getattr(_kc, "vk", None) is not None
    )
    if _PTT_MODE == "key" and _PTT_KEY is not None
    else frozenset()
)


def _ptt_key_matches(key) -> bool:
    if _PTT_MODE != "key":
        return False
    vk = getattr(key, "vk", None)
    return vk is not None and vk in _PTT_VKS


def _is_ctrl(key) -> bool:
    return key in (kb.Key.ctrl, kb.Key.ctrl_l, kb.Key.ctrl_r)


def _key_is_space(key) -> bool:
    """Space may register as Key.space or KeyCode(char=' ') depending on OS/pynput."""
    return key == kb.Key.space or getattr(key, "char", None) == " "


def _has_space() -> bool:
    return any(_key_is_space(k) for k in _pressed)


def _ptt_combo() -> bool:
    return any(_is_ctrl(k) for k in _pressed) and _has_space()


def _has_shift() -> bool:
    return any(
        k in (kb.Key.shift, kb.Key.shift_l, kb.Key.shift_r) for k in _pressed
    )


def _has_cmd() -> bool:
    return any(k in (kb.Key.cmd, kb.Key.cmd_l, kb.Key.cmd_r) for k in _pressed)


def _has_alt() -> bool:
    return any(
        k in (kb.Key.alt, kb.Key.alt_l, kb.Key.alt_r, kb.Key.alt_gr) for k in _pressed
    )


def _triple_token_active(name: str) -> bool:
    if name == "ctrl":
        return any(_is_ctrl(k) for k in _pressed)
    if name == "shift":
        return _has_shift()
    if name == "space":
        return _has_space()
    if name == "cmd":
        return _has_cmd()
    if name == "alt":
        return _has_alt()
    return False


def _ptt_triple_pressed() -> bool:
    if _PTT_MODE != "triple" or not _PTT_TRIPLE:
        return False
    return all(_triple_token_active(t) for t in _PTT_TRIPLE)


def _sanitize_text(s: str) -> str:
    """Strip NUL and other C0 controls that break terminals (^@ = Ctrl+Space / NUL)."""
    out = []
    for ch in s:
        o = ord(ch)
        if ch in "\n\t\r":
            out.append(ch)
        elif o == 0 or (o < 32 and ch not in "\n\t\r"):
            continue
        else:
            out.append(ch)
    return "".join(out).strip()


def audio_callback(indata, frames, time, status):
    if is_recording:
        with _state_lock:
            recording.append(indata.copy())


def _maybe_activate_cursor() -> None:
    """Bring Cursor to the front (optional). You still must click the chat composer for focus."""
    if os.environ.get("DICTATE_ACTIVATE_CURSOR") != "1" or platform.system() != "Darwin":
        return
    subprocess.run(
        ["osascript", "-e", 'tell application "Cursor" to activate'],
        capture_output=True,
        timeout=5,
        check=False,
    )
    time.sleep(0.2)


def _inject_paste(payload: str) -> None:
    backup = pyperclip.paste()
    try:
        pyperclip.copy(payload)
        time.sleep(0.04)
        pyautogui.hotkey("command", "v")
        time.sleep(0.03)
    finally:
        try:
            if isinstance(backup, str):
                pyperclip.copy(backup)
        except Exception:
            pass


def _copy_only(payload: str) -> None:
    """Copy payload and ask user to paste manually (works without accessibility trust)."""
    pyperclip.copy(payload)
    print("📋 Copied transcript to clipboard. Click target input and press Cmd+V.")


def _inject_untrusted_paste(payload: str, *, is_partial: bool) -> None:
    """Clipboard + Cmd+V while Accessibility is off: throttle partial pastes (default 500 ms)."""
    global _last_untrusted_paste_ts
    if platform.system() != "Darwin":
        _inject_paste(payload)
        return
    if is_partial:
        with _untrusted_paste_lock:
            now = time.time()
            if now - _last_untrusted_paste_ts < _UNTRUSTED_PASTE_INTERVAL_S:
                pyperclip.copy(payload)
                return
            _last_untrusted_paste_ts = now
    try:
        _inject_paste(payload)
    except Exception as e:
        pyperclip.copy(payload)
        print(f"⚠️ Auto-paste failed ({e!r}); copied to clipboard — Cmd+V in the target field.")


def _is_macos_trusted() -> bool:
    if platform.system() != "Darwin":
        return True
    try:
        import HIServices

        return bool(HIServices.AXIsProcessTrusted())
    except Exception:
        return False


def _extra_focus_roles() -> frozenset:
    raw = os.environ.get("DICTATE_FOCUS_EXTRA_ROLES", "")
    return frozenset(x.strip() for x in raw.split(",") if x.strip())


def _darwin_ax_focus_tags():
    """Collect AXRole / AXSubrole from focused element and ancestors.

    Returns ``(tags, err)``. ``err`` is ``None`` if ApplicationServices could not
    be loaded (caller should skip the guard). ``err == 0`` means success.
    """
    try:
        import ApplicationServices as AS
    except Exception:
        return None, None

    system = AS.AXUIElementCreateSystemWide()
    err, focused = AS.AXUIElementCopyAttributeValue(
        system, AS.kAXFocusedUIElementAttribute, None
    )
    if err != 0:
        return [], err
    if focused is None:
        return [], err

    tags = []
    el = focused
    for _ in range(16):
        for attr in (AS.kAXRoleAttribute, AS.kAXSubroleAttribute):
            e2, v = AS.AXUIElementCopyAttributeValue(el, attr, None)
            if e2 == 0 and v is not None:
                tags.append(str(v))
        e3, parent = AS.AXUIElementCopyAttributeValue(el, AS.kAXParentAttribute, None)
        if e3 != 0 or parent is None:
            break
        el = parent
    return tags, 0


def _macos_focus_allows_text_injection():
    """Return (True, None) to inject, or (False, reason)."""
    global _warned_focus_ax
    if platform.system() != "Darwin":
        return True, None
    if os.environ.get("DICTATE_FOCUS_TEXTFIELD_ONLY", "1") != "1":
        return True, None

    tags, err = _darwin_ax_focus_tags()
    extra = _extra_focus_roles()

    if err is None:
        return True, None

    if err != 0:
        ax_fail = os.environ.get("DICTATE_FOCUS_AX_FAIL", "allow").lower()
        if ax_fail == "deny":
            return (
                False,
                f"Could not read focused UI (AX error {err}). Grant Accessibility trust.",
            )
        if not _warned_focus_ax:
            print(
                "⚠️ Could not verify text-field focus (AX). Allowing injection; "
                "set DICTATE_FOCUS_AX_FAIL=deny to block until AX works."
            )
            _warned_focus_ax = True
        return True, None

    if not tags:
        return False, "Focused element has no accessible role — click a text field."

    allowed = _ALLOWED_TEXT_ROLES | extra
    if any(t in allowed for t in tags):
        return True, None

    show = " > ".join(dict.fromkeys(tags))
    return False, f"Focus is not a text field (saw: {show})."


def _inject_text(text: str, *, is_partial: bool = False) -> None:
    """Send text to whatever control is focused in the frontmost app.

    DICTATE_TYPE_MODE:
      pynput (default on macOS) — synthetic key events; usually best for Cursor/Electron.
      paste — clipboard + Cmd+V (if pynput typing misbehaves in your target app)
      clipboard — copy only (manual Cmd+V), useful before trust is granted
      write — pyautogui.write
      auto — try pynput, then paste

    When Accessibility is not trusted (Darwin): clipboard + Cmd+V; streaming partials are
    throttled to DICTATE_UNTRUSTED_PASTE_MS (default 500).

    DICTATE_FOCUS_TEXTFIELD_ONLY=1 (default on macOS): only inject when the focused
    AX role looks like a text field (see ApplicationServices). Click the target
    input first. DICTATE_FOCUS_EXTRA_ROLES adds allowed role names (comma-separated).
    """
    global _warned_untrusted_inject, _last_focus_reject_reason
    payload = text + " "
    delay_ms = int(os.environ.get("DICTATE_DELAY_MS", "0"))
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)

    _maybe_activate_cursor()

    if platform.system() == "Darwin":
        ok, reason = _macos_focus_allows_text_injection()
        if not ok:
            pyperclip.copy(payload)
            if reason != _last_focus_reject_reason:
                print(f"⛔ {reason}")
                print(
                    "   Transcript copied to clipboard — click a real text field, then Cmd+V."
                )
                _last_focus_reject_reason = reason
            return
        _last_focus_reject_reason = None

    default_mode = "pynput" if platform.system() == "Darwin" else "write"
    mode = os.environ.get("DICTATE_TYPE_MODE", default_mode).lower()
    if platform.system() == "Darwin" and not _is_macos_trusted():
        if not _warned_untrusted_inject:
            print(
                "⚠️ Not trusted yet: using clipboard + auto Cmd+V "
                f"(partials at most every {int(_UNTRUSTED_PASTE_INTERVAL_S * 1000)} ms; "
                "set DICTATE_UNTRUSTED_PASTE_MS to change)."
            )
            _warned_untrusted_inject = True
        _inject_untrusted_paste(payload, is_partial=is_partial)
        return

    if mode == "clipboard":
        _copy_only(payload)
        return

    if mode == "paste":
        if platform.system() == "Darwin":
            _inject_paste(payload)
        else:
            pyautogui.hotkey("ctrl", "v")
        return
    if mode == "write":
        pyautogui.write(payload, interval=0.02)
        return
    if mode == "pynput":
        _typer.type(payload)
        return
    # auto
    try:
        _typer.type(payload)
    except Exception:
        if platform.system() == "Darwin":
            _inject_paste(payload)
        else:
            pyautogui.write(payload, interval=0.02)


def _transcribe_chunk(audio: np.ndarray) -> str:
    if audio.size < int(fs * 0.12):
        return ""
    segments, _ = _ensure_model().transcribe(audio, vad_filter=False)
    return " ".join([seg.text for seg in segments]).strip()


def _transcribe_and_inject(audio: np.ndarray, *, partial: bool) -> None:
    text = _sanitize_text(_transcribe_chunk(audio))
    if not text:
        return
    label = "…" if partial else "📝"
    print(f"{label} {text}")
    _inject_text(text, is_partial=partial)


def _emit_stream_chunk(min_new: int, max_chunk: int) -> None:
    global last_emit_sample
    with _state_lock:
        if not recording:
            return
        full = np.concatenate(recording, axis=0).flatten().astype(np.float32)
        tail = full[last_emit_sample:]
        if tail.size < min_new:
            return
        take = min(int(tail.size), max_chunk)
        chunk = tail[:take].copy()
        last_emit_sample += take
    _transcribe_and_inject(chunk, partial=True)


def _flush_remainder() -> None:
    global last_emit_sample
    with _state_lock:
        if not recording:
            return
        full = np.concatenate(recording, axis=0).flatten().astype(np.float32)
        remainder = full[last_emit_sample:].copy()
        last_emit_sample = full.size
    if remainder.size < int(fs * 0.12):
        return
    _transcribe_and_inject(remainder, partial=False)


def _streaming_worker() -> None:
    min_new = int(STREAM_MIN_NEW_S * fs)
    max_chunk = int(STREAM_MAX_CHUNK_S * fs)
    first = True
    while True:
        time.sleep(STREAM_WAKE_S if first else STREAM_INTERVAL_S)
        first = False
        if not is_recording:
            break
        _emit_stream_chunk(min_new, max_chunk)
    _flush_remainder()


def start_recording():
    global recording, is_recording, last_emit_sample
    if is_recording:
        return
    _ensure_model()
    with _state_lock:
        recording = []
        last_emit_sample = 0
        is_recording = True
    if _PTT_MODE == "combo":
        print("🎙️ Recording…")
        return
    print("🎙️ Recording (streaming)…")
    threading.Thread(target=_streaming_worker, daemon=True).start()


def stop_recording():
    global is_recording
    if not is_recording:
        return
    is_recording = False
    print("⏹️ Finishing…")

    with _state_lock:
        empty = not recording
        audio = (
            np.concatenate(recording, axis=0).flatten().astype(np.float32)
            if not empty
            else None
        )
    if empty:
        print("(no audio captured)")
        return
    if _PTT_MODE == "combo":
        threading.Thread(
            target=_transcribe_and_inject,
            args=(audio,),
            kwargs={"partial": False},
            daemon=True,
        ).start()


def on_press(key):
    if key is None:
        return
    if os.environ.get("DICTATE_DEBUG"):
        print(f"[debug] press {key!r} vk={getattr(key, 'vk', None)}")
    try:
        _pressed.add(key)
    except TypeError:
        return

    if _PTT_MODE == "combo":
        if _ptt_combo():
            start_recording()
    elif _PTT_MODE == "triple":
        if _ptt_triple_pressed():
            start_recording()
    elif _ptt_key_matches(key):
        start_recording()


def on_release(key):
    if key is None:
        return
    if os.environ.get("DICTATE_DEBUG"):
        print(f"[debug] release {key!r} vk={getattr(key, 'vk', None)}")
    _pressed.discard(key)
    if _PTT_MODE == "combo":
        if is_recording and not _ptt_combo():
            stop_recording()
    elif _PTT_MODE == "triple":
        if is_recording and not _ptt_triple_pressed():
            stop_recording()
    elif _PTT_MODE == "key" and _ptt_key_matches(key) and is_recording:
        stop_recording()


def _open_macos_privacy_panes() -> None:
    """Best-effort: jump to Accessibility + Input Monitoring in System Settings."""
    if platform.system() != "Darwin":
        return
    urls = (
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
    )
    for u in urls:
        subprocess.run(["open", u], check=False, capture_output=True)


def _darwin_swallow_ctrl_space(event_type, event):
    """Quartz callback: drop Space while Control is held so terminals never see NUL (^@).

    pynput passes this as ``darwin_intercept``; returning None suppresses the event.
    Injected events (typing) must pass through unchanged.
    """
    from Quartz import (
        CGEventGetFlags,
        CGEventGetIntegerValueField,
        kCGEventFlagMaskControl,
        kCGEventKeyDown,
        kCGEventKeyUp,
        kCGKeyboardEventKeycode,
        kCGEventSourceUnixProcessID,
    )

    if CGEventGetIntegerValueField(event, kCGEventSourceUnixProcessID) != 0:
        return event
    if event_type not in (kCGEventKeyDown, kCGEventKeyUp):
        return event
    vk = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
    # Space — same as pynput.keyboard._darwin.Key.space
    if vk != 0x31:
        return event
    if CGEventGetFlags(event) & kCGEventFlagMaskControl:
        return None
    return event


def _darwin_intercept_for_ptt():
    """Use a modifying tap only when trusted; otherwise CGEventTapCreate would fail."""
    if platform.system() != "Darwin":
        return None
    if os.environ.get("DICTATE_SUPPRESS_CTRL_SPACE", "1") != "1":
        return None
    try:
        import HIServices

        if not HIServices.AXIsProcessTrusted():
            return None
    except Exception:
        return None
    return _darwin_swallow_ctrl_space


def _print_macos_permission_help() -> None:
    if platform.system() != "Darwin":
        return
    print()
    print("=" * 62)
    print('  macOS says "not trusted" / "accessibility clients" → BOTH must be allowed:')
    print("  • Accessibility (the error message names this — do not skip it)")
    print("  • Input Monitoring")
    print()
    print("  A) Privacy & Security → Accessibility")
    print("     → + Add Cursor.app if you run python inside Cursor’s terminal,")
    print("       OR Terminal.app if you use Terminal.")
    print("     → Also add this Python binary (toggle ON for each):")
    print(f"       {sys.executable}")
    print()
    print("  B) Privacy & Security → Input Monitoring")
    print("     → Add the SAME apps + the same python path; toggle ON.")
    print()
    print("  If an entry already exists: toggle OFF, remove it, add again, toggle ON.")
    print("  Then quit the app completely (Cmd+Q), reopen, run dictate.py again.")
    print()
    print("  Until Accessibility trusts this process, Ctrl+Space can still print ^@ (NUL)")
    print("  in the terminal — fixing trust also enables swallowing those keys.")
    print()
    print("  Optional: DICTATE_OPEN_SETTINGS=1 opens Accessibility + Input Monitoring panes.")
    print("=" * 62)
    print()
    if os.environ.get("DICTATE_OPEN_SETTINGS") == "1":
        print("Opening System Settings privacy panes…")
        _open_macos_privacy_panes()
        print()


# Start mic + keyboard before loading Whisper (model loads on first PTT)
if platform.system() == "Darwin":
    _print_macos_permission_help()

stream = sd.InputStream(
    callback=audio_callback,
    samplerate=fs,
    channels=1,
    blocksize=1024,
)
stream.start()

_darwin_ic = _darwin_intercept_for_ptt()
_listener_kw = {}
if _darwin_ic is not None:
    _listener_kw["darwin_intercept"] = _darwin_ic

listener = kb.Listener(on_press=on_press, on_release=on_release, **_listener_kw)
listener.start()

_LABELS = {"ctrl": "Ctrl", "shift": "Shift", "space": "Space", "cmd": "Cmd", "alt": "Alt"}


def _ptt_hint_text() -> str:
    if _PTT_MODE == "combo":
        return "Ctrl+Space"
    if _PTT_MODE == "triple" and _PTT_TRIPLE:
        return "+".join(_LABELS.get(t, t) for t in _PTT_TRIPLE)
    return f"hold {os.environ.get('DICTATE_PTT', _DEFAULT_PTT).upper()}"


print("Listening for keyboard… (DICTATE_DEBUG=1 prints each key)")
if _PTT_MODE == "key" and _PTT_VKS:
    print(f"  PTT accepts these key codes (vk): {sorted(_PTT_VKS)}")

print(f"PTT: hold {_ptt_hint_text()}  (DICTATE_PTT=ctrl_space | ctrl_shift_space | f7 | …)")
if _PTT_MODE == "combo":
    print(
        "  Ctrl+Space: record while held; text is typed once you release (no streaming while Ctrl is down)."
    )
if platform.system() == "Darwin" and _darwin_ic is None:
    print(
        "  ^@ fix: grant Accessibility trust for this Python binary, then restart — "
        "Ctrl+Space is swallowed so the terminal won’t get NUL. "
        "(Or set DICTATE_SUPPRESS_CTRL_SPACE=0 to disable.)"
    )
print("Release PTT to finish.")
if _PTT_MODE == "triple":
    print("  If the terminal shows garbage while dictating: DICTATE_TYPE_MODE=paste")
if platform.system() == "Darwin":
    print()
    print("Accessibility + Input Monitoring for Cursor/Terminal/python; click target field to type.")
    if os.environ.get("DICTATE_FOCUS_TEXTFIELD_ONLY", "1") == "1":
        print(
            "  Text-field guard: injection only when a text field has keyboard focus (AX). "
            "DICTATE_FOCUS_TEXTFIELD_ONLY=0 disables; DICTATE_FOCUS_EXTRA_ROLES adds roles."
        )
    if _PTT_MODE == "key":
        print("F-row: enable “Use F1–F12 as standard function keys” if fn+F-keys don’t register.")
    print()
try:
    listener.join()
except KeyboardInterrupt:
    listener.stop()
