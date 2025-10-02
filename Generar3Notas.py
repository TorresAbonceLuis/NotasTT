# three_notes.py
import argparse, subprocess
from pathlib import Path
import numpy as np
import pretty_midi
import soundfile as sf

# -------------------- utilidades --------------------
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def name_to_midi(name: str) -> int:
    """Convierte C4, F#3, etc. a número MIDI."""
    name = name.strip().upper().replace('DB','C#').replace('EB','D#').replace('GB','F#').replace('AB','G#').replace('BB','A#')
    # separa letra(s) y octava
    for i in range(len(name)):
        if name[i] in "-0123456789":
            pitch, octv = name[:i], int(name[i:])
            break
    else:
        raise ValueError(f"Nota inválida: {name}")
    if pitch not in NOTE_NAMES:
        raise ValueError(f"Nota inválida: {name}")
    return (octv + 1) * 12 + NOTE_NAMES.index(pitch)

def render_with_fluidsynth(sf2_path, mid_path, wav_path, fs=44100, gain="2.0"):
    cmd = ["fluidsynth", "-r", str(fs), "-F", str(wav_path), "-g", str(gain), "-ni", str(sf2_path), str(mid_path)]
    subprocess.run(cmd, check=True)

def normalize_audio(wav_path, target_peak=0.9):
    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if x.ndim > 1: x = x.mean(axis=1)
    peak = float(np.max(np.abs(x)) + 1e-12)
    g = target_peak / peak
    x = np.clip(x * g, -1.0, 1.0)
    sf.write(str(wav_path), x, sr)

# -------------------- programa --------------------
def main():
    ap = argparse.ArgumentParser(description="Genera un WAV con 3 notas separadas por 0.5 s usando un .sf2")
    ap.add_argument("--sf2", type=str, default="piano.sf2", help="Ruta al SoundFont .sf2")
    ap.add_argument("--notes", type=str, default="C4,E4,G4", help="Tres notas separadas por coma (ej: C4,E4,G4)")
    ap.add_argument("--dur", type=float, default=1.0, help="Duración de cada nota en segundos")
    ap.add_argument("--gap", type=float, default=0.5, help="Silencio entre notas (segundos)")
    ap.add_argument("--vel", type=int, default=90, help="Velocity (1–127)")
    ap.add_argument("--pedal", action="store_true", help="Sostiene pedal (CC64) durante cada nota")
    ap.add_argument("--lead", type=float, default=0.25, help="Silencio inicial (s)")
    ap.add_argument("--fs", type=int, default=44100, help="Sample rate WAV")
    ap.add_argument("--out_dir", type=Path, default=Path("demo_out"))
    ap.add_argument("--basename", type=str, default="three_notes")
    ap.add_argument("--normalize", action="store_true", help="Normaliza a -0.9 dBFS aprox")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mid_path = args.out_dir / f"{args.basename}.mid"
    wav_path = args.out_dir / f"{args.basename}.wav"

    # Parsear notas
    note_tokens = [s.strip() for s in args.notes.split(",") if s.strip()]
    if len(note_tokens) != 3:
        raise SystemExit("Debes proporcionar EXACTAMENTE 3 notas (ej: --notes C4,E4,G4)")
    notes_midi = [name_to_midi(n) for n in note_tokens]

    # Construir MIDI
    pm = pretty_midi.PrettyMIDI(resolution=480)
    inst = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    t = float(args.lead)
    for i, midi_note in enumerate(notes_midi):
        start = t
        end   = t + float(args.dur)
        if args.pedal:
            inst.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=start))
        inst.notes.append(pretty_midi.Note(velocity=int(args.vel), pitch=int(midi_note), start=start, end=end))
        if args.pedal:
            inst.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=end))
        # avanzar tiempo: nota + gap
        t = end + float(args.gap)

    pm.instruments.append(inst)
    pm.write(str(mid_path))

    # Renderizar con Fluidsynth
    render_with_fluidsynth(args.sf2, mid_path, wav_path, fs=args.fs, gain="2.0")

    # Normalizar (opcional)
    if args.normalize:
        normalize_audio(wav_path, target_peak=0.9)

    print("✅ Generado:")
    print(" MIDI:", mid_path)
    print(" WAV :", wav_path)

if __name__ == "__main__":
    main()
