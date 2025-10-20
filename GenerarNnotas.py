# GenerarNnotas.py
import argparse, subprocess, random
from pathlib import Path
import numpy as np
import pretty_midi
import soundfile as sf

# -------------------- utilidades --------------------
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def midi_to_name(midi_num: int) -> str:
    """Convierte número MIDI a nombre de nota (ej: 60 -> C4)"""
    octave = (midi_num // 12) - 1
    note = NOTE_NAMES[midi_num % 12]
    return f"{note}{octave}"

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
    """Renderiza MIDI a WAV usando FluidSynth"""
    cmd = ["fluidsynth", "-r", str(fs), "-F", str(wav_path), "-g", str(gain), "-ni", str(sf2_path), str(mid_path)]
    subprocess.run(cmd, check=True)

def normalize_audio(wav_path, target_peak=0.9):
    """Normaliza el audio al peak objetivo"""
    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if x.ndim > 1: x = x.mean(axis=1)
    peak = float(np.max(np.abs(x)) + 1e-12)
    g = target_peak / peak
    x = np.clip(x * g, -1.0, 1.0)
    sf.write(str(wav_path), x, sr)

# -------------------- programa --------------------
def main():
    ap = argparse.ArgumentParser(description="Genera un WAV con N notas aleatorias separadas por 0.5 s usando un .sf2")
    ap.add_argument("-n", "--num_notes", type=int, required=True, help="Número de notas a generar")
    ap.add_argument("--sf2", type=str, default="piano.sf2", help="Ruta al SoundFont .sf2")
    ap.add_argument("--min_midi", type=int, default=21, help="Nota MIDI mínima (default: 21 = A0)")
    ap.add_argument("--max_midi", type=int, default=108, help="Nota MIDI máxima (default: 108 = C8)")
    ap.add_argument("--dur", type=float, default=1.0, help="Duración de cada nota en segundos")
    ap.add_argument("--gap", type=float, default=0.5, help="Silencio entre notas (segundos)")
    ap.add_argument("--vel", type=int, default=90, help="Velocity (1–127)")
    ap.add_argument("--pedal", action="store_true", help="Sostiene pedal (CC64) durante cada nota")
    ap.add_argument("--lead", type=float, default=0.25, help="Silencio inicial (s)")
    ap.add_argument("--fs", type=int, default=44100, help="Sample rate WAV")
    ap.add_argument("--out_dir", type=Path, default=Path("demo_out"))
    ap.add_argument("--basename", type=str, default=None, help="Nombre base del archivo (default: n_notes_random)")
    ap.add_argument("--normalize", action="store_true", help="Normaliza a -0.9 dBFS aprox")
    ap.add_argument("--seed", type=int, default=None, help="Semilla para reproducibilidad")
    ap.add_argument("--unique", action="store_true", help="Garantiza que no se repitan notas")
    args = ap.parse_args()

    # Validaciones
    if args.num_notes < 1:
        raise SystemExit("❌ El número de notas debe ser al menos 1")
    
    if args.min_midi < 0 or args.max_midi > 127:
        raise SystemExit("❌ Rango MIDI debe estar entre 0 y 127")
    
    if args.min_midi >= args.max_midi:
        raise SystemExit("❌ min_midi debe ser menor que max_midi")
    
    available_notes = args.max_midi - args.min_midi + 1
    if args.unique and args.num_notes > available_notes:
        raise SystemExit(f"❌ No hay suficientes notas únicas en el rango [{args.min_midi}, {args.max_midi}]. "
                        f"Disponibles: {available_notes}, solicitadas: {args.num_notes}")

    # Configurar semilla
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Usando semilla: {args.seed}")

    # Generar notas aleatorias
    midi_range = list(range(args.min_midi, args.max_midi + 1))
    if args.unique:
        notes_midi = random.sample(midi_range, args.num_notes)
    else:
        notes_midi = random.choices(midi_range, k=args.num_notes)
    
    note_names = [midi_to_name(m) for m in notes_midi]
    
    # Preparar directorios y nombres
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.basename is None:
        args.basename = f"{args.num_notes}_notes_random"
    
    mid_path = args.out_dir / f"{args.basename}.mid"
    wav_path = args.out_dir / f"{args.basename}.wav"

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
    print(f"🎹 Generando {args.num_notes} notas aleatorias...")
    render_with_fluidsynth(args.sf2, mid_path, wav_path, fs=args.fs, gain="2.0")

    # Normalizar (opcional)
    if args.normalize:
        normalize_audio(wav_path, target_peak=0.9)
        print("🔊 Audio normalizado")

    print("✅ Generado:")
    print(f"   MIDI: {mid_path}")
    print(f"   WAV : {wav_path}")
    print(f"   Notas ({args.num_notes}): {', '.join(note_names)}")
    print(f"   Duración total: {t:.2f} segundos")

if __name__ == "__main__":
    main()
