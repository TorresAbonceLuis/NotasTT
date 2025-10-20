import subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf

# CONFIGURACIÓN FASE 2 - DATASET ROBUSTO
SOUNDFONT = "piano.sf2"
OUT_ROOT = Path("data/piano")
META_CSV = Path("metadata/index.csv")

FS = 44100
BITS = 16
LEAD_SIL = 0.2
TAIL_SIL = 0.1

# VARIABILIDAD COMPLETA
VELOCITIES = [30, 60, 90, 110]  # 4 velocidades
ARTICULATIONS = {
    "staccato": 0.3, 
    "portato": 0.8, 
    "sustain": 1.5,
    "legato": 2.0
}  # 4 articulaciones
PEDALS = {"noped": 0, "ped": 127}  # 2 opciones
TAKES = [1]  # 1 toma

# RANGO DE NOTAS
MIDI_MIN, MIDI_MAX = 21, 108  # A0 a C8

def midi_to_name(m):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note = names[m % 12]
    octave = (m // 12) - 1
    return f"{note}{octave}"

def write_midi_phase2(note_midi, velocity, dur, pedal_val, out_mid_path):
    """MIDI con expresividad mejorada"""
    pm = pretty_midi.PrettyMIDI(resolution=480)
    inst = pretty_midi.Instrument(program=0)
    
    # Pedal al inicio si está activado
    if pedal_val > 0:
        inst.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=0.0))
    
    # Nota con timing natural
    start = LEAD_SIL
    end = LEAD_SIL + float(dur)
    inst.notes.append(pretty_midi.Note(
        velocity=velocity, 
        pitch=note_midi, 
        start=start, 
        end=end
    ))
    
    # Liberar pedal al final si estaba activado
    if pedal_val > 0:
        inst.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=end))
    
    pm.instruments.append(inst)
    pm.write(str(out_mid_path))

def render_phase2(sf2_path, mid_path, wav_path, fs=44100):
    """Renderizado optimizado para calidad"""
    cmd = [
        "fluidsynth",
        "-r", str(fs),
        "-F", str(wav_path),
        "-g", "1.2",  # Ganancia más conservadora
        "-C", "0",    # Sin chorus
        "-R", "0",    # Sin reverb
        "-z", "2048", # Buffer más grande para estabilidad
        "-O", "float", # Máxima calidad
        "-ni", str(sf2_path), str(mid_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"❌ Error en render: {e}")
        return False

def robust_silence_trim(wav_path, note_midi):
    """Recorte robusto adaptado a cada nota"""
    try:
        data, sr = sf.read(str(wav_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        
        # Threshold adaptativo por rango de notas
        if note_midi < 60:  # Graves
            threshold_db = -35
            tail_duration = 0.15
        elif note_midi < 72:  # Medias
            threshold_db = -40  
            tail_duration = 0.12
        else:  # Agudas
            threshold_db = -45
            tail_duration = 0.08
        
        # Calcular envelope en dB
        envelope = 20 * np.log10(np.abs(data) + 1e-12)
        
        # Encontrar regiones no silenciosas
        non_silent = envelope > threshold_db
        
        # Suavizar detección
        window_size = int(0.02 * sr)  # 20ms
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(non_silent.astype(float), kernel, mode='same') > 0.1
        
        # Encontrar inicio y fin
        nonzero_indices = np.where(smoothed)[0]
        if len(nonzero_indices) == 0:
            return False
            
        start_idx = max(0, nonzero_indices[0] - int(0.03 * sr))
        end_idx = min(len(data), nonzero_indices[-1] + int(tail_duration * sr))
        
        # Verificar duración mínima
        min_samples = int(0.25 * sr)  # 250ms mínimo
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

def normalize_phase2(wav_path):
    """Normalización inteligente"""
    try:
        data, sr = sf.read(str(wav_path))
        
        # RMS normalization
        rms = np.sqrt(np.mean(data**2))
        if rms < 1e-8:
            return False
            
        target_rms = 0.15  # -16 dBFS aprox
        gain = target_rms / rms
        
        # Soft clipping para evitar distorsión
        normalized = np.tanh(data * gain * 0.7) * 1.1
        normalized = np.clip(normalized, -0.98, 0.98)
        
        sf.write(str(wav_path), normalized, sr)
        return True
        
    except Exception as e:
        print(f"❌ Error normalizando: {e}")
        return False

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    META_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    
    total_files = (MIDI_MAX - MIDI_MIN + 1) * len(VELOCITIES) * len(ARTICULATIONS) * len(PEDALS) * len(TAKES)
    print("🚀 FASE 2: GENERANDO DATASET ROBUSTO PARA NOTAS INDIVIDUALES")
    print("=" * 60)
    print(f"🎯 Objetivo: {total_files} archivos")
    print(f"📊 Rango: {midi_to_name(MIDI_MIN)} a {midi_to_name(MIDI_MAX)} ({MIDI_MAX - MIDI_MIN + 1} notas)")
    print(f"🎹 Variabilidad:")
    print(f"   • Velocidades: {len(VELOCITIES)}")
    print(f"   • Articulaciones: {len(ARTICULATIONS)}") 
    print(f"   • Pedal: {len(PEDALS)}")
    print(f"   • Tomas: {len(TAKES)}")
    
    current_file = 0
    start_time = time.time()
    failed_files = []

    for midi_note in range(MIDI_MIN, MIDI_MAX + 1):
        note_name = midi_to_name(midi_note)
        note_dir = OUT_ROOT / note_name
        note_dir.mkdir(parents=True, exist_ok=True)

        for vel in VELOCITIES:
            for art_name, dur in ARTICULATIONS.items():
                for pedal_name, pedal_val in PEDALS.items():
                    for take in TAKES:
                        current_file += 1
                        
                        base = f"{note_name}_v{vel}_{art_name}_{pedal_name}_t{take}"
                        mid_path = note_dir / f"{base}.mid"
                        wav_path = note_dir / f"{base}.wav"
                        
                        # Progreso cada 50 archivos
                        if current_file % 50 == 0 or current_file <= 10:
                            elapsed = time.time() - start_time
                            # SOLUCIÓN: Evitar división por cero
                            if elapsed > 0:
                                rate = current_file / elapsed
                                eta_seconds = (total_files - current_file) / rate
                                eta_minutes = eta_seconds / 60
                                print(f"[{current_file:4d}/{total_files}] {base} - {rate:.1f} files/sec - ETA: {eta_minutes:.1f}min")
                            else:
                                # Si elapsed es 0, mostrar información básica sin tasa
                                print(f"[{current_file:4d}/{total_files}] {base} - procesando...")

                        max_attempts = 2
                        success = False
                        
                        for attempt in range(max_attempts):
                            try:
                                # Generar MIDI
                                write_midi_phase2(midi_note, vel, dur, pedal_val, mid_path)
                                
                                # Renderizar
                                if not render_phase2(SOUNDFONT, mid_path, wav_path, FS):
                                    continue
                                
                                # Post-procesamiento
                                if not normalize_phase2(wav_path):
                                    continue
                                    
                                if not robust_silence_trim(wav_path, midi_note):
                                    print(f"  ⚠️  Problema recortando {base}")
                                    # Continuar de todos modos
                                
                                success = True
                                break
                                
                            except Exception as e:
                                print(f"  ❌ Intento {attempt + 1} falló: {e}")
                                continue

                        if not success:
                            print(f"  💥 Falló después de {max_attempts} intentos: {base}")
                            failed_files.append(base)
                            continue

                        # Medir duración final y metadata
                        try:
                            data, sr = sf.read(str(wav_path))
                            final_duration = len(data) / sr
                            
                            rows.append({
                                "filepath": str(wav_path),
                                "note": note_name,
                                "midi": midi_note,
                                "velocity": vel,
                                "articulation": art_name,
                                "pedal": 1 if pedal_val > 0 else 0,
                                "duration": round(final_duration, 3),
                                "take": take
                            })
                            
                            print(f"  ✅ {base}: {final_duration:.2f}s")
                            
                        except Exception as e:
                            print(f"  ❌ Error obteniendo metadata: {e}")
                            failed_files.append(base)

    # Guardar metadata
    df = pd.DataFrame(rows)
    df.to_csv(META_CSV, index=False)
    
    # Reporte final
    print(f"\n🎉 FASE 2 COMPLETADA")
    print("=" * 40)
    print(f"✅ Archivos exitosos: {len(df)}/{total_files}")
    print(f"❌ Archivos fallidos: {len(failed_files)}")
    
    if failed_files:
        print(f"📝 Fallos guardados en: failed_phase2.txt")
        with open("failed_phase2.txt", "w") as f:
            for file in failed_files:
                f.write(file + "\n")
    
    # Estadísticas
    if len(df) > 0:
        avg_dur = df['duration'].mean()
        min_dur = df['duration'].min()
        max_dur = df['duration'].max()
        
        print(f"\n📊 Estadísticas:")
        print(f"   • Duración promedio: {avg_dur:.2f}s")
        print(f"   • Rango: {min_dur:.2f}s - {max_dur:.2f}s")
        print(f"   • Notas únicas: {df['note'].nunique()}")
        print(f"   • Velocidades únicas: {df['velocity'].nunique()}")
        print(f"   • Articulaciones: {df['articulation'].nunique()}")
    
    print(f"\n💾 Metadata guardada en: {META_CSV}")
    print("🎯 Próximo paso: Crear splits y entrenar modelo")

if __name__ == "__main__":
    main()