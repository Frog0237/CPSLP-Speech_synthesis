from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import re
import numpy as np
import nltk
from nltk.corpus import cmudict

import simpleaudio
from synth_args import process_commandline


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z']+")

_CMU_DICT: Optional[dict] = None


def _get_cmudict() -> dict:
    global _CMU_DICT
    if _CMU_DICT is not None:
        return _CMU_DICT
    try:
        _CMU_DICT = cmudict.dict()
        return _CMU_DICT
    except LookupError:
        nltk.download("cmudict", quiet=True)
        _CMU_DICT = cmudict.dict()
        return _CMU_DICT


def _strip_stress(phone: str) -> str:
    return re.sub(r"\d", "", phone).lower()


def _validate_reverse_mode(mode: Optional[str]) -> None:
    valid = ("words", "phones", "signal", None)
    if mode not in valid:
        raise ValueError(f"Invalid reverse mode: {mode}. Must be one of {list(valid)}")


def _validate_volume(volume: Optional[int]) -> None:
    if volume is None:
        return
    if not isinstance(volume, int) or not (0 <= volume <= 100):
        raise ValueError("Volume must be between 0 and 100")


def _apply_signal_reverse(audio: simpleaudio.Audio, reverse_mode: Optional[str]) -> None:
    if reverse_mode == "signal":
        audio.data = audio.data[::-1].copy()


def _apply_volume(audio: simpleaudio.Audio, volume: Optional[int]) -> None:
    if volume is None:
        return
    _validate_volume(volume)
    audio.rescale(volume / 100.0)


def _make_silence_like(template: simpleaudio.Audio, seconds: float) -> simpleaudio.Audio:
    silence = simpleaudio.Audio()
    silence.chan = template.chan
    silence.rate = template.rate
    silence.format = template.format
    silence.nptype = template.nptype
    n = max(0, int(template.rate * seconds))
    silence.data = np.zeros(n, dtype=template.data.dtype)
    return silence


def _concat_with_silence(audio_list: Sequence[simpleaudio.Audio], silence_seconds: float) -> simpleaudio.Audio:
    if not audio_list:
        out = simpleaudio.Audio()
        out.data = np.array([], dtype=np.int16)
        return out

    first = audio_list[0]
    silence = np.zeros(int(first.rate * silence_seconds), dtype=first.data.dtype)

    segments: List[np.ndarray] = []
    for i, a in enumerate(audio_list):
        if i > 0:
            segments.append(silence)
        segments.append(a.data)

    combined = simpleaudio.Audio()
    combined.chan = first.chan
    combined.rate = first.rate
    combined.format = first.format
    combined.nptype = first.nptype
    combined.data = np.concatenate(segments) if segments else np.array([], dtype=first.data.dtype)
    return combined


def _split_sentences(text: str) -> List[str]:
    t = text.replace("\n", " ").strip()
    if not t:
        return []
    parts = _SENTENCE_SPLIT_RE.split(t)
    return [p.strip() for p in parts if p.strip()]


class Synth:
    def __init__(self, wav_folder):
        self.diphones = self.get_wavs(wav_folder)

    def get_wavs(self, wav_folder):
        wav_path = Path(wav_folder)
        if not wav_path.is_dir():
            raise FileNotFoundError(f"Diphone folder not found: {wav_folder}")

        diphones: Dict[str, simpleaudio.Audio] = {}
        for wav_file in wav_path.glob("*.wav"):
            unit_name = wav_file.stem.lower()
            audio = simpleaudio.Audio()
            audio.load(str(wav_file))
            diphones[unit_name] = audio

        if not diphones:
            raise FileNotFoundError(f"No wav files found in folder: {wav_folder}")

        return diphones

    def phones_to_diphones(self, phones):
        if len(phones) < 2:
            raise ValueError("Not enough phones to form a diphone (at least 2)")

        diphone_names: List[str] = []
        for p1, p2 in zip(phones[:-1], phones[1:]):
            name = f"{p1}-{p2}"
            if name not in self.diphones:
                raise ValueError(f"Diphone '{name}' not found in dictionary")
            diphone_names.append(name)

        return diphone_names

    @staticmethod
    def _init_audio_like(template: simpleaudio.Audio) -> simpleaudio.Audio:
        out = simpleaudio.Audio()
        out.chan = template.chan
        out.rate = template.rate
        out.format = template.format
        out.nptype = template.nptype
        return out

    @staticmethod
    def _crossfade_concat(a: np.ndarray, b: np.ndarray, overlap: int) -> np.ndarray:
        if overlap <= 0:
            return np.concatenate([a, b])

        ov = min(overlap, a.shape[0], b.shape[0])
        if ov <= 0:
            return np.concatenate([a, b])

        tail = a[-ov:].astype(np.float64)
        head = b[:ov].astype(np.float64)

        window = np.hanning(2 * ov)
        fade_in = window[:ov]
        fade_out = window[ov:]

        mixed = tail * fade_out + head * fade_in

        if np.issubdtype(a.dtype, np.integer):
            info = np.iinfo(a.dtype)
            mixed = np.clip(mixed, info.min, info.max)

        out = a.copy()
        out[-ov:] = mixed.astype(a.dtype)
        return np.concatenate([out, b[ov:]])

    def synthesise(self, phones, reverse=False, smooth_concat=False):
        diphone_names = self.phones_to_diphones(phones)

        if not diphone_names:
            out = simpleaudio.Audio()
            out.data = np.array([], dtype=np.int16)
            return out

        first_audio = self.diphones[diphone_names[0]]
        out = self._init_audio_like(first_audio)

        out_data = first_audio.data.copy()

        if smooth_concat:
            overlap = int(first_audio.rate * 0.010)
            if overlap < 1:
                overlap = 1
        else:
            overlap = 0

        for name in diphone_names[1:]:
            next_data = self.diphones[name].data
            if smooth_concat and overlap > 0:
                out_data = self._crossfade_concat(out_data, next_data, overlap)
            else:
                out_data = np.concatenate([out_data, next_data])

        out.data = out_data.astype(out.nptype)
        return out


def load_addpron_file(path):
    extra_lex: Dict[str, List[str]] = {}

    addpron_path = Path(path)
    if not addpron_path.is_file():
        raise FileNotFoundError(f"Addendum file not found: {path}")

    with addpron_path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Line {line_num} in addendum file is not long enough (at least 2 parts)")

            word = parts[0].lower()
            phones = [_strip_stress(p) for p in parts[1:]]
            extra_lex[word] = phones

    return extra_lex


class Utterance:
    def __init__(self, phrase, extra_lex=None):
        self.phrase = phrase
        self.extra_lex = {} if extra_lex is None else extra_lex

    def _tokenize(self, text):
        text = text.lower()
        return _WORD_RE.findall(text)

    def get_phone_seq(self, reverse=None):
        words = self._tokenize(self.phrase)
        if not words:
            raise ValueError("No words found in the phrase")

        if reverse == "words":
            words = list(reversed(words))

        lex = _get_cmudict()
        phones: List[str] = []

        for w in words:
            if w in self.extra_lex:
                word_phones = self.extra_lex[w]
            else:
                if w not in lex:
                    raise ValueError(f"Word '{w}' not found in dictionary")
                word_phones = [_strip_stress(sym) for sym in lex[w][0]]

            phones.extend(word_phones)

        full_phones = ["pau"] + phones + ["pau"]

        if reverse == "phones":
            full_phones = list(reversed(full_phones))

        return full_phones


def process_file(textfile, args):
    audio_list: List[simpleaudio.Audio] = []

    text_path = Path(textfile)
    if not text_path.is_file():
        raise FileNotFoundError(f"Text file not found: {textfile}")

    extra_lex = {}
    if hasattr(args, "addpron") and args.addpron is not None:
        extra_lex = load_addpron_file(args.addpron)

    synth = Synth(wav_folder=args.diphones)

    with text_path.open("r", encoding="utf-8") as f:
        raw_text = f.read()

    sentences = _split_sentences(raw_text)
    for sent in sentences:
        utt = Utterance(phrase=sent, extra_lex=extra_lex)
        if args.reverse in ("phones", "words"):
            phone_seq = utt.get_phone_seq(reverse=args.reverse)
        else:
            phone_seq = utt.get_phone_seq(reverse=None)

        audio_out = synth.synthesise(phone_seq, smooth_concat=args.crossfade)
        audio_list.append(audio_out)

    return audio_list


def main(args):
    _validate_reverse_mode(args.reverse)

    if hasattr(args, "fromfile") and args.fromfile is not None:
        audio_list = process_file(args.fromfile, args)

        if not audio_list:
            raise ValueError("No audio generated from file")

        for audio in audio_list:
            _apply_signal_reverse(audio, args.reverse)
            _apply_volume(audio, args.volume)

        if args.outfile is not None:
            combined = _concat_with_silence(audio_list, silence_seconds=0.4)
            combined.save(args.outfile)

        if args.play:
            silence = _make_silence_like(audio_list[0], seconds=0.4)
            for i, audio in enumerate(audio_list):
                audio.play()
                if i < len(audio_list) - 1:
                    silence.play()
        return

    if args.phrase is None:
        raise ValueError("No phrase given")

    synth = Synth(wav_folder=args.diphones)

    extra_lex = {}
    if hasattr(args, "addpron") and args.addpron is not None:
        extra_lex = load_addpron_file(args.addpron)

    utt = Utterance(phrase=args.phrase, extra_lex=extra_lex)

    if args.reverse in ("phones", "words"):
        phone_seq = utt.get_phone_seq(reverse=args.reverse)
    else:
        phone_seq = utt.get_phone_seq(reverse=None)

    audio_out = synth.synthesise(phone_seq, smooth_concat=args.crossfade)

    _apply_signal_reverse(audio_out, args.reverse)
    _apply_volume(audio_out, args.volume)

    if args.outfile is not None:
        audio_out.save(args.outfile)

    if args.play:
        audio_out.play()


if __name__ == "__main__":
    main(process_commandline())
