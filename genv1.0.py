#!/usr/bin/env python3
import os
import random
import mido
from collections import defaultdict, Counter

INPUT_DIR = 'midi' #директория с midi файлами
OUTPUT_DIR = 'audio' #директория с выходным midi файлом
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'generated_clean.mid')

MAX_GLOBAL_POLY = 5      # максимум одновременных нот во всех каналах
MAX_CHANNELS = 5         # сколько каналов использовать одновременно максимально 
GRID = 240               # размер шага в тиках (например при 480 tpb -> 1/8 ноты)
BARS = 32                # длина генерируемой секции в тактах
ONSET_PROB = 0.55        # вероятность начать ноту на шаге для выбранного канала


def load_midis(folder):
    mids = []
    if not os.path.isdir(folder):
        return mids
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith('.mid') or name.lower().endswith('.midi'):
            path = os.path.join(folder, name)
            try:
                mids.append(mido.MidiFile(path))
                print("Loaded:", name)
            except Exception as e:
                print("Failed to load", name, ":", e)
    return mids


def extract_sequences(mid_files):
    seqs = defaultdict(list)
    programs = {}
    ticks_per_beat = 480
    tempo = 500000
    for mid in mid_files:
        ticks_per_beat = mid.ticks_per_beat or ticks_per_beat
        events = []
        for tr in mid.tracks:
            abs_t = 0
            for msg in tr:
                abs_t += msg.time
                events.append((abs_t, msg.copy()))
        events.sort(key=lambda x: x[0])

        active = defaultdict(dict)
        for abs_t, msg in events:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            if msg.type == 'program_change':
                programs[msg.channel] = msg.program
            if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
                active[msg.channel][msg.note] = abs_t
            elif msg.type == 'note_off' or (msg.type == 'note_on' and getattr(msg, 'velocity', 0) == 0):
                ch = msg.channel
                n = msg.note
                if ch in active and n in active[ch]:
                    start = active[ch].pop(n)
                    dur = abs_t - start
                    if dur <= 0:
                        dur = ticks_per_beat // 4
                    seqs[ch].append((n, dur))
    return seqs, programs, ticks_per_beat, tempo


def build_markov_models(seqs):
    models = {}
    for ch, arr in seqs.items():
        trans = defaultdict(Counter)
        for i in range(len(arr) - 1):
            cur = arr[i][0]
            nxt = arr[i + 1][0]
            trans[cur][nxt] += 1
        models[ch] = trans
    return models


PREFER_STEP_WEIGHT = 1.2
SCALE_PENALTY = 0.35


def choose_note(trans_counter, prev_note, fallback_notes):
    if trans_counter:
        candidates = list(trans_counter.items())
        weights = []
        for note, cnt in candidates:
            interval = abs(note - prev_note) if prev_note is not None else 0
            lead = 1.0 if interval == 0 else (1.0 / (1 + (interval / 7.0) ** PREFER_STEP_WEIGHT))
            w = cnt * lead
            weights.append((note, w))
        total = sum(w for _, w in weights)
        if total <= 0:
            return candidates[0][0]
        r = random.random() * total
        s = 0.0
        for n, w in weights:
            s += w
            if r <= s:
                return n
        return weights[-1][0]
    if fallback_notes:
        return random.choice(fallback_notes)
    return 60


def generate_events(models, seqs, programs, tpb, tempo):
    channel_activity = sorted(seqs.items(), key=lambda kv: -len(kv[1]))
    channels = [ch for ch, _ in channel_activity[:MAX_CHANNELS]]
    if not channels:
        channels = [0, 1, 2, 3]

    print("Using channels:", channels)

    total_ticks = BARS * 4 * tpb
    steps = int(total_ticks // GRID)

    events_by_ch = defaultdict(list)
    active_until = {}
    last_note = {}
    fallback_notes = {ch: [n for n, _ in seqs[ch]] for ch in seqs}

    for step in range(steps):
        abs_tick = step * GRID
        global_active = sum(1 for end in active_until.values() if end > abs_tick)
        ch = channels[step % len(channels)]
        ch_busy = active_until.get(ch, 0) > abs_tick

        if (not ch_busy) and (global_active < MAX_GLOBAL_POLY) and (random.random() < ONSET_PROB):
            if global_active >= (MAX_GLOBAL_POLY - 1):
                dur_steps = random.choices([1, 2], weights=[0.9, 0.1])[0]
            else:
                dur_steps = random.choices([1, 2, 4], weights=[0.6, 0.3, 0.1])[0]
            dur_ticks = dur_steps * GRID

            model = models.get(ch, {})
            prev = last_note.get(ch, None)
            note = choose_note(model.get(prev, Counter()), prev, fallback_notes.get(ch, []))
            last_note[ch] = note

            on_msg = mido.Message('note_on', note=int(note), velocity=96, channel=ch, time=0)
            off_msg = mido.Message('note_off', note=int(note), velocity=0, channel=ch, time=0)
            events_by_ch[ch].append((abs_tick, on_msg))
            events_by_ch[ch].append((abs_tick + dur_ticks, off_msg))
            active_until[ch] = abs_tick + dur_ticks

    return events_by_ch, channels


def write_midi(events_by_ch, channels, programs, tpb, tempo, out_path):
    mid = mido.MidiFile(ticks_per_beat=tpb)
    for ch in sorted(channels):
        tr = mido.MidiTrack()
        prog = programs.get(ch, 6)
        tr.append(mido.Message('program_change', program=prog, channel=ch, time=0))
        evs = events_by_ch.get(ch, [])
        if not evs:
            mid.tracks.append(tr)
            continue
        evs.sort(key=lambda x: x[0])
        prev_tick = 0
        for atick, msg in evs:
            delta = atick - prev_tick
            msg.time = max(0, int(delta))
            tr.append(msg)
            prev_tick = atick
        mid.tracks.append(tr)

    if mid.tracks:
        mid.tracks[0].insert(0, mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mid.save(out_path)
    print("Saved:", out_path)


def main():
    mids = load_midis(INPUT_DIR)
    if not mids:
        print("No MIDI files in", INPUT_DIR)
        return
    seqs, programs, tpb, tempo = extract_sequences(mids)
    print("Channels found (activity):", {ch: len(v) for ch, v in seqs.items()})
    models = build_markov_models(seqs)
    events_by_ch, channels = generate_events(models, seqs, programs, tpb, tempo)
    write_midi(events_by_ch, channels, programs, tpb, tempo, OUTPUT_FILE)


if __name__ == '__main__':
    main()