import os, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf

SOUNDFONT = "piano.sf2"#piano
OUT_ROOT = Path("data/piano_A")#salida de audios
META_CSV = Path("metadata/index.csv")#salida de metadatos

FS = 44100#sample rate WAV
BITS= 16#bits WAV (16 por defecto)
LEAD_SIL = 0.25#silencio antes de la nota (s)
TAIL_SIL = 0.25#silencio después (s)
# Variantes por nota
VELOCITIES = [30, 60, 90, 110]#velocidades
ARTICULATIONS = {"staccato": 0.30, "sustain": 1.50}
PEDALS = {"noped": 0, "ped": 127}#pedales
TAKES = [1]#tomas

MIDI_MIN, MIDI_MAX = 21, 108  # A0..C8

def midi_to_name(m):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note = names[m % 12]
    octave = (m // 12) - 1
    return f"{note}{octave}"

def write_midi(note_midi, velocity, dur, pedal_val, out_mid_path):
    pm = pretty_midi.PrettyMIDI(resolution=480)
    inst = pretty_midi.Instrument(program=0)
    if pedal_val > 0:
        inst.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=0.0))
    start = LEAD_SIL
    end = LEAD_SIL + float(dur)
    inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=note_midi, start=start, end=end))
    if pedal_val > 0:
        inst.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=end))
    pm.instruments.append(inst)
    pm.write(str(out_mid_path))

def render_with_fluidsynth(sf2_path, mid_path, wav_path, fs=44100):
    cmd = [
        "fluidsynth",
        "-r", str(fs),
        "-F", str(wav_path),
        "-ni", str(sf2_path), str(mid_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def peak_dbfs(wav_path):
    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.max(np.abs(x)) + 1e-12
    return 20*np.log10(peak)

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    META_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for midi_note in range(MIDI_MIN, MIDI_MAX+1):
        note_name = midi_to_name(midi_note)
        note_dir = OUT_ROOT / note_name
        note_dir.mkdir(parents=True, exist_ok=True)

        for vel in VELOCITIES:
            for art_name, dur in ARTICULATIONS.items():
                for pedal_name, pedal_val in PEDALS.items():
                    for take in TAKES:
                        base = f"{note_name}_v{vel}_{art_name}_{pedal_name}_{take:02d}"
                        mid_path = note_dir / f"{base}.mid"
                        wav_path = note_dir / f"{base}.wav"

                        write_midi(midi_note, vel, dur, pedal_val, mid_path)
                        render_with_fluidsynth(SOUNDFONT, mid_path, wav_path, fs=FS)

                        pk = peak_dbfs(wav_path)

                        rows.append({
                            "filepath": str(wav_path),
                            "instrument": "piano_A",
                            "note": note_name,
                            "midi": midi_note,
                            "velocity": vel,
                            "articulation": art_name,
                            "pedal": 1 if pedal_val>0 else 0,
                            "seconds": round(LEAD_SIL + dur + TAIL_SIL, 3),
                            "fs": FS,
                            "bits": BITS,
                            "peak_dbfs": round(pk, 2),
                            "source": "synth_sf2",
                            "soundfont": os.path.basename(SOUNDFONT),
                        })

    df = pd.DataFrame(rows)
    df.to_csv(META_CSV, index=False)
    print(f"Listo. Escribí {len(rows)} archivos y el índice en {META_CSV}")

if __name__ == "__main__":
    main()