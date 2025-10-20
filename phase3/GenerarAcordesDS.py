# generate_chords_3notes.py
import os, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf
from scipy import signal

# CONFIGURACIÓN PARA ACORDES DE 3 NOTAS
SOUNDFONT = "../piano.sf2"
OUT_ROOT = Path("data/piano")
META_CSV = Path("metadata/index.csv")

FS = 44100
BITS = 16
LEAD_SIL = 0.2
TAIL_SIL = 0.1

# CONFIGURACIÓN BALANCEADA
VELOCITIES = [60, 90]           # 2 velocidades
ARTICULATIONS = {"sustain": 2.0} # Mayor duración para acordes
PEDALS = {"ped": 127}           # Con pedal para sonido más natural
TAKES = [1]                     # 1 toma

# ACORDES MÁS COMUNES DE 3 NOTAS (TRÍADAS)
CHORDS = {
    # Acordes mayores (20 acordes)
    "C_major": [60, 64, 67],      # C4, E4, G4
    "G_major": [67, 71, 74],      # G4, B4, D5
    "D_major": [62, 66, 69],      # D4, F#4, A4
    "A_major": [69, 73, 76],      # A4, C#5, E5
    "E_major": [64, 68, 71],      # E4, G#4, B4
    "F_major": [65, 69, 72],      # F4, A4, C5
    "Bb_major": [70, 74, 77],     # Bb4, D5, F5
    "Eb_major": [63, 67, 70],     # Eb4, G4, Bb4
    "Ab_major": [68, 72, 75],     # Ab4, C5, Eb5
    "Db_major": [61, 65, 68],     # Db4, F4, Ab4
    
    # Acordes menores (15 acordes)
    "A_minor": [69, 72, 76],      # A4, C5, E5
    "E_minor": [64, 67, 71],      # E4, G4, B4
    "D_minor": [62, 65, 69],      # D4, F4, A4
    "G_minor": [67, 70, 74],      # G4, Bb4, D5
    "C_minor": [60, 63, 67],      # C4, Eb4, G4
    "F_minor": [65, 68, 72],      # F4, Ab4, C5
    "Bb_minor": [70, 73, 77],     # Bb4, Db5, F5
    "Eb_minor": [63, 66, 70],     # Eb4, Gb4, Bb4
    
    # Acordes suspendidos (10 acordes)
    "Csus2": [60, 62, 67],        # C4, D4, G4
    "Csus4": [60, 65, 67],        # C4, F4, G4
    "Gsus2": [67, 69, 74],        # G4, A4, D5
    "Gsus4": [67, 72, 74],        # G4, C5, D5
    "Dsus2": [62, 64, 69],        # D4, E4, A4
    "Dsus4": [62, 67, 69],        # D4, G4, A4
    "Asus2": [69, 71, 76],        # A4, B4, E5
    "Asus4": [69, 74, 76],        # A4, D5, E5
    
    # Acordes aumentados y disminuidos (5 acordes)
    "C_aug": [60, 64, 68],        # C4, E4, G#4
    "C_dim": [60, 63, 66],        # C4, Eb4, Gb4
    "G_aug": [67, 71, 75],        # G4, B4, D#5
    "G_dim": [67, 70, 73],        # G4, Bb4, Db5
}

def chord_to_name(chord_notes):
    """Convierte notas MIDI a nombre de acorde legible"""
    base_note = chord_notes[0]
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    base_name = names[base_note % 12]
    octave = (base_note // 12) - 1
    return f"{base_name}{octave}"

def write_midi_chord(chord_notes, velocity, dur, pedal_val, out_mid_path):
    """Escribe archivo MIDI para acorde de 3 notas"""
    pm = pretty_midi.PrettyMIDI(resolution=480)
    inst = pretty_midi.Instrument(program=0)
    
    # Pedal al inicio
    if pedal_val > 0:
        inst.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=0.0))
    
    # Las 3 notas del acorde
    start = LEAD_SIL
    end = LEAD_SIL + float(dur)
    
    for note in chord_notes:
        inst.notes.append(pretty_midi.Note(
            velocity=velocity, 
            pitch=note, 
            start=start, 
            end=end
        ))
    
    # Liberar pedal al final
    if pedal_val > 0:
        inst.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=end))
    
    pm.instruments.append(inst)
    pm.write(str(out_mid_path))

def render_chord(sf2_path, mid_path, wav_path, fs=44100):
    """Renderizado optimizado para acordes"""
    cmd = [
        "fluidsynth",
        "-r", str(fs),
        "-F", str(wav_path),
        "-g", "1.0",  # Ganancia más conservadora para acordes
        "-C", "0",
        "-R", "0", 
        "-z", "2048",
        "-O", "float",
        "-ni", str(sf2_path), str(mid_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"❌ Error en render: {e}")
        return False

def adaptive_silence_trim_chord(wav_path, chord_notes):
    """Recorte adaptativo específico para acordes"""
    try:
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        # Usar la nota más grave del acorde para determinar parámetros
        lowest_note = min(chord_notes)
        
        if lowest_note < 48:  # Acordes graves
            threshold_db = -30
            tail_duration = 0.3
            min_duration = 0.5
        elif lowest_note < 72:  # Acordes medios
            threshold_db = -35
            tail_duration = 0.2
            min_duration = 0.4
        else:  # Acordes agudos
            threshold_db = -40
            tail_duration = 0.15
            min_duration = 0.3
        
        # Calcular envelope
        envelope = 20 * np.log10(np.abs(data) + 1e-12)
        
        # Encontrar regiones no silenciosas
        non_silent = envelope > threshold_db
        
        # Suavizar detección
        window_size = int(0.05 * sr)
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(non_silent.astype(float), kernel, mode='same') > 0.1
        
        nonzero_indices = np.where(smoothed)[0]
        if len(nonzero_indices) == 0:
            return False
            
        start_idx = max(0, nonzero_indices[0] - int(0.1 * sr))
        end_idx = min(len(data), nonzero_indices[-1] + int(tail_duration * sr))
        
        # Verificar duración mínima
        min_samples = int(min_duration * sr)
        if (end_idx - start_idx) < min_samples:
            return False
        
        trimmed = data[start_idx:end_idx]
        sf.write(str(wav_path), trimmed, sr)
        
        original_dur = len(data) / sr
        new_dur = len(trimmed) / sr
        print(f"  ✂️  {original_dur:.2f}s → {new_dur:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en recorte: {e}")
        return False

def normalize_chord(wav_path):
    """Normalización optimizada para acordes"""
    try:
        data, sr = sf.read(str(wav_path))
        
        rms = np.sqrt(np.mean(data**2))
        if rms < 1e-8:
            return False
            
        target_rms = 0.1  # Más conservador para acordes
        gain = target_rms / rms
        
        # Compresión más suave para acordes
        normalized = np.tanh(data * gain * 0.6) * 1.1
        normalized = np.clip(normalized, -0.95, 0.95)
        
        sf.write(str(wav_path), normalized, sr)
        return True
        
    except Exception as e:
        print(f"❌ Error normalizando: {e}")
        return False

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    META_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    
    total_files = len(CHORDS) * len(VELOCITIES) * len(ARTICULATIONS) * len(PEDALS) * len(TAKES)
    
    print("🎹 GENERANDO ACORDES DE 3 NOTAS (TRÍADAS)")
    print("=" * 60)
    print(f"🎯 Total de acordes: {len(CHORDS)}")
    print(f"📊 Total de archivos: {total_files}")
    print(f"🎹 Configuración:")
    print(f"   • Velocidades: {VELOCITIES}")
    print(f"   • Articulación: {list(ARTICULATIONS.keys())[0]}")
    print(f"   • Pedal: Sí")
    print(f"   • Tomas: 1")
    
    current_file = 0
    start_time = time.time()
    failed_files = []

    # Estadísticas por tipo de acorde
    chord_type_stats = {"major": 0, "minor": 0, "sus": 0, "aug_dim": 0}

    for chord_name, chord_notes in CHORDS.items():
        # Clasificar acorde para estadísticas
        if "major" in chord_name and "minor" not in chord_name:
            chord_type = "major"
        elif "minor" in chord_name:
            chord_type = "minor"
        elif "sus" in chord_name:
            chord_type = "sus"
        else:
            chord_type = "aug_dim"
        
        chord_dir = OUT_ROOT / chord_name
        chord_dir.mkdir(parents=True, exist_ok=True)

        for vel in VELOCITIES:
            for art_name, dur in ARTICULATIONS.items():
                for pedal_name, pedal_val in PEDALS.items():
                    for take in TAKES:
                        current_file += 1
                        chord_type_stats[chord_type] += 1
                        
                        base = f"{chord_name}_v{vel}"
                        mid_path = chord_dir / f"{base}.mid"
                        wav_path = chord_dir / f"{base}.wav"
                        
                        # Progreso
                        if current_file % 10 == 0 or current_file <= 5:
                            elapsed = time.time() - start_time
                            rate = current_file / elapsed if elapsed > 0 else 0
                            eta_seconds = (total_files - current_file) / rate if rate > 0 else 0
                            eta_minutes = eta_seconds / 60
                            print(f"[{current_file:2d}/{total_files}] {chord_name} v{vel} - {rate:.1f} files/sec")

                        max_attempts = 2
                        success = False
                        
                        for attempt in range(max_attempts):
                            try:
                                # Generar MIDI para acorde
                                write_midi_chord(chord_notes, vel, dur, pedal_val, mid_path)
                                
                                # Renderizar
                                if not render_chord(SOUNDFONT, mid_path, wav_path, FS):
                                    continue
                                
                                # Post-procesamiento
                                if not normalize_chord(wav_path):
                                    continue
                                    
                                if not adaptive_silence_trim_chord(wav_path, chord_notes):
                                    print(f"  ⚠️  Problema recortando {chord_name}")
                                
                                success = True
                                break
                                
                            except Exception as e:
                                print(f"  ❌ Intento {attempt + 1} falló: {e}")
                                continue

                        if not success:
                            print(f"  💥 Falló después de {max_attempts} intentos: {chord_name}")
                            failed_files.append(chord_name)
                            continue

                        # Metadata enriquecida
                        try:
                            data, sr = sf.read(str(wav_path))
                            final_duration = len(data) / sr
                            
                            # Convertir notas a nombres
                            note_names = [f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][n % 12]}{(n // 12) - 1}" for n in chord_notes]
                            
                            rows.append({
                                "filepath": str(wav_path),
                                "chord_name": chord_name,
                                "chord_type": chord_type,
                                "notes_midi": str(chord_notes),
                                "notes_names": str(note_names),
                                "base_note": chord_notes[0],
                                "base_note_name": note_names[0],
                                "velocity": vel,
                                "articulation": art_name,
                                "pedal": 1 if pedal_val > 0 else 0,
                                "duration": round(final_duration, 3),
                                "num_notes": len(chord_notes)
                            })
                            
                            print(f"  ✅ {chord_name}: {final_duration:.2f}s - {', '.join(note_names)}")
                            
                        except Exception as e:
                            print(f"  ❌ Error en metadata: {e}")
                            failed_files.append(chord_name)

    # Guardar metadata
    df = pd.DataFrame(rows)
    df.to_csv(META_CSV, index=False)
    
    # Reporte final
    print(f"\n🎉 GENERACIÓN DE ACORDES COMPLETADA")
    print("=" * 50)
    print(f"✅ Acordes exitosos: {len(df)}/{total_files}")
    print(f"❌ Acordes fallidos: {len(failed_files)}")
    
    if failed_files:
        with open("failed_chords.txt", "w") as f:
            for file in failed_files:
                f.write(file + "\n")
        print(f"📝 Fallos guardados en: failed_chords.txt")
    
    # Estadísticas detalladas
    if len(df) > 0:
        print(f"\n📊 ESTADÍSTICAS DETALLADAS:")
        print(f"   • Duración promedio: {df['duration'].mean():.2f}s")
        print(f"   • Acordes generados: {df['chord_name'].nunique()}/{len(CHORDS)}")
        print(f"   • Notas por acorde: 3 (siempre)")
        
        print(f"\n🎵 DISTRIBUCIÓN POR TIPO:")
        for chord_type, count in chord_type_stats.items():
            if count > 0:
                print(f"   • {chord_type.capitalize()}: {count} archivos")
        
        # Ejemplos de acordes
        print(f"\n🎹 EJEMPLOS DE ACORDES:")
        sample_chords = df.sample(min(5, len(df)))
        for _, chord in sample_chords.iterrows():
            print(f"   • {chord['chord_name']}: {chord['notes_names']}")
    
    print(f"\n💾 Metadata guardada en: {META_CSV}")
    print("🚀 Próximo paso: python make_splits_fixed.py --meta_csv metadata/chords_3notes_index.csv")

if __name__ == "__main__":
    main()