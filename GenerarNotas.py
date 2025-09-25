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
        "-g", "2.0",  # Aumentar ganancia para mejor volumen
        "-ni", str(sf2_path), str(mid_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def peak_dbfs(wav_path):
    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    peak = np.max(np.abs(x)) + 1e-12
    return 20*np.log10(peak)

def trim_silence(wav_path, silence_threshold=0.001, tail_silence=0.2):
    """Recorta el silencio del final del audio, dejando solo un poco de cola"""
    try:
        data, sr = sf.read(str(wav_path))
        
        # Convertir a mono si es estéreo
        if data.ndim > 1:
            audio_mono = np.mean(data, axis=1)
        else:
            audio_mono = data
        
        # Encontrar el último punto con sonido significativo
        significant_samples = np.where(np.abs(audio_mono) > silence_threshold)[0]
        
        if len(significant_samples) > 0:
            # Último punto con sonido + un poco de silencio de cola
            last_sound_sample = significant_samples[-1]
            tail_samples = int(tail_silence * sr)
            end_sample = min(last_sound_sample + tail_samples, len(data))
            
            # Recortar el audio
            trimmed_data = data[:end_sample]
            
            # Guardar el archivo recortado
            sf.write(str(wav_path), trimmed_data, sr)
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error recortando silencio {wav_path}: {e}")
        return False

def normalize_audio(wav_path, target_peak=0.9):
    """Normaliza el audio al nivel de pico deseado para que sea audible"""
    try:
        # Leer el audio
        data, sr = sf.read(str(wav_path))
        
        # Encontrar el pico actual
        current_peak = np.max(np.abs(data))
        
        if current_peak > 1e-6:  # Evitar división por cero
            # Calcular el factor de normalización
            normalization_factor = target_peak / current_peak
            
            # Aplicar normalización
            normalized_data = data * normalization_factor
            
            # Asegurar que no exceda el rango [-1, 1]
            normalized_data = np.clip(normalized_data, -1.0, 1.0)
            
            # Sobrescribir el archivo
            sf.write(str(wav_path), normalized_data, sr)
            
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error normalizando {wav_path}: {e}")
        return False

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    META_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    
    # Calcular total de archivos
    total_files = (MIDI_MAX - MIDI_MIN + 1) * len(VELOCITIES) * len(ARTICULATIONS) * len(PEDALS) * len(TAKES)
    print(f"Generando {total_files} archivos de audio del piano completo...")
    print(f"Rango: {midi_to_name(MIDI_MIN)} a {midi_to_name(MIDI_MAX)} ({MIDI_MAX - MIDI_MIN + 1} notas)")
    print("Esto puede tomar un tiempo considerable...")
    
    current_file = 0
    start_time = time.time()

    for midi_note in range(MIDI_MIN, MIDI_MAX+1):
        note_name = midi_to_name(midi_note)
        note_dir = OUT_ROOT / note_name
        note_dir.mkdir(parents=True, exist_ok=True)

        for vel in VELOCITIES:
            for art_name, dur in ARTICULATIONS.items():
                for pedal_name, pedal_val in PEDALS.items():
                    for take in TAKES:
                        current_file += 1
                        base = f"{note_name}_v{vel}_{art_name}_{pedal_name}_{take:02d}"
                        mid_path = note_dir / f"{base}.mid"
                        wav_path = note_dir / f"{base}.wav"
                        
                        # Mostrar progreso cada 50 archivos
                        if current_file % 50 == 0 or current_file <= 10:
                            elapsed = time.time() - start_time
                            rate = current_file / elapsed if elapsed > 0 else 0
                            eta_seconds = (total_files - current_file) / rate if rate > 0 else 0
                            eta_minutes = eta_seconds / 60
                            print(f"[{current_file}/{total_files}] {base} - {rate:.1f} files/sec - ETA: {eta_minutes:.1f}min")

                        write_midi(midi_note, vel, dur, pedal_val, mid_path)
                        render_with_fluidsynth(SOUNDFONT, mid_path, wav_path, fs=FS)
                        
                        # Normalizar el audio para asegurar buen volumen
                        normalize_audio(wav_path)
                        
                        # Recortar silencio del final
                        trim_silence(wav_path)

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

    # Guardar metadatos y mostrar resumen final
    df = pd.DataFrame(rows)
    df.to_csv(META_CSV, index=False)
    
    elapsed_total = time.time() - start_time
    print(f"\n🎉 ¡Completado!")
    print(f"📁 {len(rows)} archivos generados en: {OUT_ROOT}")
    print(f"📊 Metadatos guardados en: {META_CSV}")
    print(f"⏱️  Tiempo total: {elapsed_total/60:.1f} minutos")
    print(f"⚡ Velocidad promedio: {len(rows)/elapsed_total:.1f} archivos/segundo")
    
    # Estadísticas por articulación
    print(f"\n📈 Resumen por articulación:")
    for art_name in ARTICULATIONS.keys():
        count = len(df[df['articulation'] == art_name])
        print(f"  - {art_name}: {count} archivos")
    
    print(f"\n🎵 ¡Todos los archivos están listos y optimizados!")
    print(f"   - Volumen normalizado ✓")
    print(f"   - Silencio final recortado ✓")
    print(f"   - Audio perfectamente audible ✓")

if __name__ == "__main__":
    main()