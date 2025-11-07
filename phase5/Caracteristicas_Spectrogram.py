# -*- coding: utf-8 -*-
# Caracteristicas_Spectrogram.py
#
# !! VERSIÓN CNN (Plan B) !!
#
# Este script implementa la estrategia de "Onsets and Frames" pero
# genera Mel Spectrograms (en lugar de Wavelets) como características 'X',
# que es la entrada estándar para los modelos CNN+LSTM.
#
# 1. Procesa el audio (X) como un Mel Spectrogram (HOP=512).
# 2. Genera 'Y_frames': un pianoroll binario (0/1) de notas activas (con pedal).
# 3. Genera 'Y_onsets': un pianoroll binario (0/1) solo de "ataques" (sin pedal).

import os, json, math, csv, traceback
from typing import Tuple, List, Dict
import numpy as np
import librosa  # Usamos Librosa para los Mel Spectrograms
# import pywt  <-- ¡Ya no se necesita!
import pretty_midi as pm
import time

# ========= RUTAS (AJUSTA AQUÍ) =========
# Lee de la misma carpeta de audio filtrado
SONGS_DIR = r"2004" 

# !! NUEVA CARPETA DE SALIDA !!
# Guarda los nuevos NPZ en una carpeta separada
OUT_DIR   = r"Training\adaptive_features_Spectrogram"
SUMMARY_CSV = os.path.join(OUT_DIR, "songs_audit_summary_Spectrogram.csv")

# ========= PARÁMETROS DE AUDIO / MEL SPECTROGRAM =========
SR = 22050
HOP_LENGTH = 512  # ¡CRUCIAL! Define nuestra resolución temporal

# Parámetros del Espectrograma (para la "imagen" 2D)
N_FFT = 2048      # Tamaño de la ventana FFT (resolución de frecuencia)
N_MELS = 128      # Número de "píxeles" en el eje Y (altura de la imagen)
F_MIN = 25.0      # Frecuencia mínima (basada en nuestro filtro)
F_MAX = 6000.0    # Frecuencia máxima (basada en nuestro filtro)


# ========= MIDI / ETIQUETAS (Sin Cambios) =========
LOW_MIDI  = 21   # A0
HIGH_MIDI = 108  # C8
N_KEYS    = HIGH_MIDI - LOW_MIDI + 1

USE_SUSTAIN_PEDAL = True
CC_SUSTAIN        = 64
PEDAL_ON_VALUE    = 64

# ========= CONTROL (Sin Cambios) =========
RESUME_IF_NPZ_EXISTS = False # Forzar re-procesamiento
EPS = 1e-12

# ========= UTILIDADES (Solo 'load_audio_mono' se mantiene) =========
def load_audio_mono(path: str, sr: int = SR):
    y, sr = librosa.load(path, sr=sr, mono=True)
    peak = np.max(np.abs(y)) if y.size else 1.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32), sr

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! NUEVA FUNCIÓN DE EXTRACCIÓN DE CARACTERÍSTICAS !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def extract_mel_spectrogram(y: np.ndarray, sr: int):
    """
    Calcula el Mel Spectrogram logarítmico (en dB)
    y lo alinea para que tenga la forma (n_frames, n_mels)
    """
    # 1. Calcular el Mel Spectrogram
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX
    )
    
    # 2. Convertir a Decibelios (escala logarítmica)
    # Esto es estándar para que las CNNs funcionen mejor
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # 3. Transponer (girar) la matriz
    # La salida de Librosa es (n_mels, n_frames)
    # La entrada de Keras (LSTM) espera (n_frames, n_features)
    # Así que la giramos a (n_frames, n_mels)
    X = S_db.T
    
    # 4. Generar los tiempos de frame (para alinear con el MIDI)
    frame_times = librosa.frames_to_time(
        np.arange(X.shape[0]), 
        sr=sr, 
        hop_length=HOP_LENGTH
    )
    
    # Normalizar los datos (rango -1 a 1 o 0 a 1)
    # Dividimos por 80.0 (un valor estándar para dB)
    X = (X / 80.0) + 1.0 # Rango aprox [0, 1]
    
    return X.astype(np.float32), frame_times
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! FIN DE LA NUEVA FUNCIÓN !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def collect_notes_with_pedal(pm_obj: pm.PrettyMIDI,
                                use_pedal=True, cc_num=64, pedal_on_value=64):
    # Esta función es idéntica a la anterior
    notes = []
    for inst in pm_obj.instruments:
        if inst.is_drum:
            continue
        sustain = []
        if use_pedal:
            vals = [(cc.time, cc.value) for cc in inst.control_changes if cc.number == cc_num]
            vals.sort(key=lambda x: x[0])
            on_time = None
            for t, v in vals:
                if v >= pedal_on_value and on_time is None:
                    on_time = t
                elif v < pedal_on_value and on_time is not None:
                    sustain.append((on_time, t))
                    on_time = None
            if on_time is not None:
                sustain.append((on_time, pm_obj.get_end_time()))
        def extend(t_end: float) -> float:
            if not use_pedal or not sustain:
                return t_end
            for a, b in sustain:
                if a <= t_end <= b:
                    return b
            return t_end
        for n in inst.notes:
            s, e = float(n.start), extend(float(n.end))
            if e > s:
                notes.append((int(n.pitch), s, e))
    notes.sort(key=lambda x: (x[0], x[1], x[2]))
    merged = []
    for p, s, e in notes:
        if not merged or merged[-1][0] != p or s > merged[-1][2] + 1e-4:
            merged.append([p, s, e])
        else:
            merged[-1][2] = max(merged[-1][2], e)
    return [(p, s, e) for p, s, e in merged]

def build_labels_onsets_frames(mid_path: str, frame_times: np.ndarray):
    # Esta función es idéntica a la anterior
    nF = len(frame_times)
    hop_t = (HOP_LENGTH / SR)
    
    Y_frames = np.zeros((nF, N_KEYS), dtype=np.uint8)
    Y_onsets = np.zeros((nF, N_KEYS), dtype=np.uint8)
    
    try:
        midi = pm.PrettyMIDI(mid_path)
    except Exception as e:
        print(f"    ERROR: Falla al cargar MIDI {mid_path}. {e}")
        return Y_frames, Y_onsets, None

    # --- PASO 1: Calcular Y_frames (Notas Activas, CON pedal) ---
    notes_with_pedal = collect_notes_with_pedal(midi, use_pedal=USE_SUSTAIN_PEDAL,
                                                cc_num=CC_SUSTAIN, pedal_on_value=PEDAL_ON_VALUE)
    
    for pitch, start, end in notes_with_pedal:
        if pitch < LOW_MIDI or pitch > HIGH_MIDI:
            continue
        k = pitch - LOW_MIDI
        s_idx = max(0, int(np.floor(start / hop_t)))
        e_idx = min(nF, int(np.ceil(end  / hop_t)))
        if e_idx > s_idx:
            Y_frames[s_idx:e_idx, k] = 1 

    # --- PASO 2: Calcular Y_onsets (Ataques, SIN pedal) ---
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes: 
            pitch = int(note.pitch)
            if pitch < LOW_MIDI or pitch > HIGH_MIDI:
                continue
            k = pitch - LOW_MIDI
            s_idx = max(0, int(np.floor(note.start / hop_t)))
            if s_idx < nF:
                Y_onsets[s_idx, k] = 1 

    return Y_frames, Y_onsets, midi

def write_summary_row(base: str,
                        m: pm.PrettyMIDI,
                        X: np.ndarray,
                        Y_frames: np.ndarray,
                        Y_onsets: np.ndarray,
                        frame_times: np.ndarray,
                        wav_path: str,
                        midi_path: str):
    
    if m is None: return
    os.makedirs(OUT_DIR, exist_ok=True)
    nF = len(frame_times)
    
    poly = (Y_frames > 0).sum(axis=1) 
    mean_poly = float(np.mean(poly))
    dur_s = float(nF * HOP_LENGTH) / SR
    notes_total = sum(len(i.notes) for i in m.instruments)
    try:
        pitch_min = min(n.pitch for i in m.instruments for n in i.notes)
        pitch_max = max(n.pitch for i in m.instruments for n in i.notes)
    except ValueError:
        pitch_min, pitch_max = "", ""
    
    poly_hist = {k: int((poly == k).sum()) for k in range(6)}
    poly_hist[5] += int((poly > 5).sum())

    row = {
        "wav_file": os.path.basename(wav_path),
        "midi_file": os.path.basename(midi_path),
        "out_file": f"{base}_Spectrogram_OnsetsFrames.npz", # Nuevo nombre
        "duration_s": round(dur_s, 3),
        "frames": int(nF),
        "features_per_frame": int(X.shape[1]), # Ahora será N_MELS (128)
        "label_type": "OnsetsFrames", 
        "hop_length": int(HOP_LENGTH),
        "feature_type": "MelSpectrogram", # Nuevo campo
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        
        "midi_notes_total": int(notes_total),
        "midi_pitch_min": pitch_min,
        "midi_pitch_max": pitch_max,
        "mean_poly": round(mean_poly, 3),
        
        "total_onset_frames": int((Y_onsets > 0).sum()),
        "total_active_frames": int((Y_frames > 0).sum()),
        
        "poly0_frames": poly_hist[0],
        "poly1_frames": poly_hist[1],
        "poly2_frames": poly_hist[2],
        "poly3_frames": poly_hist[3],
        "poly4_frames": poly_hist[4],
        "poly5plus_frames": poly_hist[5],
    }

    header = list(row.keys())
    write_header = not os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(row)

def save_npz(base: str,
                X: np.ndarray, 
                Y_frames: np.ndarray, 
                Y_onsets: np.ndarray, 
                frame_times: np.ndarray,
                wav_path: str, midi_path: str):
                
    out_npz = os.path.join(OUT_DIR, f"{base}_Spectrogram_OnsetsFrames.npz")
    meta = dict(
        wav_file=os.path.basename(wav_path), mid_file=os.path.basename(midi_path),
        sr=SR, hop_length=HOP_LENGTH,
        feature_type="MelSpectrogram",
        n_mels=N_MELS, n_fft=N_FFT, f_min=F_MIN, f_max=F_MAX,
        low_midi=LOW_MIDI, high_midi=HIGH_MIDI, n_keys=N_KEYS,
        label_type="OnsetsFrames"
    )
    
    np.savez_compressed(
        out_npz,
        X=X.astype(np.float32),
        Y_frames=Y_frames.astype(np.uint8),
        Y_onsets=Y_onsets.astype(np.uint8),
        frame_times=frame_times.astype(np.float32),
        meta=np.array([json.dumps(meta)], dtype=object)
    )
    return out_npz

def find_song_pairs(songs_dir: str):
    # Sin cambios
    files = os.listdir(songs_dir)
    wavs = {}
    midis = {}
    for f in files:
        p = os.path.join(songs_dir, f)
        if os.path.isdir(p):
            continue
        name, ext = os.path.splitext(f)
        ext_l = ext.lower()
        if ext_l == ".wav":
            wavs[name.lower()] = p
        elif ext_l in (".mid", ".midi"):
            midis[name.lower()] = p
    bases = sorted(set(wavs.keys()) & set(midis.keys()))
    pairs = [(b, wavs[b], midis[b]) for b in bases]
    missing_wav = sorted(set(midis.keys()) - set(wavs.keys()))
    missing_mid = sorted(set(wavs.keys()) - set(midis.keys()))
    return pairs, missing_wav, missing_mid

def process_one_song(base: str, wav_path: str, midi_path: str):
    
    if RESUME_IF_NPZ_EXISTS:
        out_npz = os.path.join(OUT_DIR, f"{base}_Spectrogram_OnsetsFrames.npz")
        if os.path.exists(out_npz):
            print(f"⏭️  SKIP (existe): {os.path.basename(out_npz)}")
            return True

    # 1. Audio → X (¡NUEVA FUNCIÓN!)
    y, sr = load_audio_mono(wav_path, sr=SR)
    X, frame_times = extract_mel_spectrogram(y, sr) # <--- ¡CAMBIO!

    # 2. MIDI → Y_frames, Y_onsets (Función Antigua)
    Y_frames, Y_onsets, m = build_labels_onsets_frames(midi_path, frame_times)

    if m is None:
        print(f"❌ ERROR en {base}: Falla al procesar el MIDI. Se saltará.")
        return False

    assert X.shape[0] == Y_frames.shape[0] == Y_onsets.shape[0] == len(frame_times)

    # 3. Recortar Silencios (Lógica Antigua)
    active_frames_indices = np.where(Y_frames.sum(axis=1) > 0)[0]
    
    X_trimmed, Y_frames_trimmed, Y_onsets_trimmed, times_trimmed = X, Y_frames, Y_onsets, frame_times
    original_frames = len(X)
    
    if len(active_frames_indices) > 0:
        start_idx = active_frames_indices[0]
        end_idx = active_frames_indices[-1]
        
        X_trimmed = X[start_idx : end_idx + 1]
        Y_frames_trimmed = Y_frames[start_idx : end_idx + 1]
        Y_onsets_trimmed = Y_onsets[start_idx : end_idx + 1]
        times_trimmed = frame_times[start_idx : end_idx + 1]
        
        print(f"    Recortando silencios: {original_frames} frames -> {len(X_trimmed)} frames")
    else:
        print("    ADVERTENCIA: No se encontraron notas activas en el MIDI. No se recortará.")

    # 4. Guardar (Funciones actualizadas)
    write_summary_row(base, m, X_trimmed, Y_frames_trimmed, Y_onsets_trimmed, times_trimmed, wav_path, midi_path)
    out_npz = save_npz(base, X_trimmed, Y_frames_trimmed, Y_onsets_trimmed, times_trimmed, wav_path, midi_path)

    # 5. Log
    poly = (Y_frames_trimmed > 0).sum(axis=1)
    mean_poly = float(np.mean(poly))
    dur_s = len(X_trimmed) * (HOP_LENGTH / SR)
    print(f"✅ {base}: frames={len(X_trimmed)}  feat/frame={X_trimmed.shape[1]}  dur={dur_s:.1f}s  mean_poly={mean_poly:.2f}")
    print(f"    → guardado: {out_npz}")
    return True

# ========= MAIN (con cronómetro) =========
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    
    script_start_time = time.time()

    pairs, missing_wav, missing_mid = find_song_pairs(SONGS_DIR)
    print(f"Iniciando PROCESO 'Spectrogram (Onsets & Frames)' para {len(pairs)} pares WAV+MIDI en: {SONGS_DIR}")
    if missing_wav:
        print("⚠️ MIDI sin WAV:", [f + ".mid/.midi" for f in missing_wav])
    if missing_mid:
        print("⚠️ WAV sin MIDI:", [f + ".wav" for f in missing_mid])

    ok, fail = 0, 0
    for base, wav_path, midi_path in pairs:
        try:
            done = process_one_song(base, wav_path, midi_path)
            ok += int(bool(done))
        except Exception as e:
            fail += 1
            print(f"❌ ERROR en {base}: {e}")
            traceback.print_exc()

    script_end_time = time.time()
    total_time = script_end_time - script_start_time

    print("\n===== RESUMEN LOTE (Spectrogram - Onsets & Frames) =====")
    print(f"Procesadas OK: {ok}  |  Fallidas: {fail}")
    print(f"CSV resumen: {SUMMARY_CSV}")
    print(f"NPZ dir: {OUT_DIR}")
    print(f"\n⏰ TIEMPO TOTAL: {total_time:.2f} segundos")