# infer_note_optimized.py
import argparse, json, pickle
from pathlib import Path
import numpy as np
import soundfile as sf
import pywt
import pretty_midi
from scipy import signal
from scipy.ndimage import median_filter

MIDI_MIN, MIDI_MAX = 21, 108
DEFAULTS = dict(fs=44100, frame_sec=0.050, hop_sec=0.025, wavelet="db8", level=9)

class WPTProcessor:
    """Procesador optimizado de Wavelet Packet Transform para inferencia"""
    
    def __init__(self, wavelet="db8", level=9, fs=44100):
        self.wavelet = wavelet
        self.level = level
        self.fs = fs
        self.n_bands = 2 ** level
        self.band_bw = (fs / 2.0) / self.n_bands
        self._overlap_matrix = None
        self._precompute_overlap_matrix()
    
    def _precompute_overlap_matrix(self):
        """Precalcula la matriz de superposición entre bandas WPT y teclas del piano"""
        B = self.n_bands
        f_low = np.arange(B) * self.band_bw
        f_high = (np.arange(B) + 1) * self.band_bw
        
        overlap_matrix = np.zeros((B, 88), dtype=np.float64)
        
        for b in range(B):
            bl, bh = f_low[b], f_high[b]
            if bh <= KEY_EDGES[0] or bl >= KEY_EDGES[-1]:
                continue
                
            i_start = np.searchsorted(KEY_EDGES, bl, side='right') - 1
            i_end = np.searchsorted(KEY_EDGES, bh, side='left')
            i_start = max(i_start, 0)
            i_end = min(i_end, 88)
            
            if i_end <= i_start:
                continue
                
            for i in range(i_start, i_end):
                kl, kh = KEY_EDGES[i], KEY_EDGES[i+1]
                overlap = max(0.0, min(bh, kh) - max(bl, kl))
                if overlap > 0:
                    overlap_matrix[b, i] = overlap / (bh - bl)
        
        self._overlap_matrix = overlap_matrix
    
    def compute_band_energies_batch(self, frames_batch):
        """Calcula energías de bandas WPT para un lote de frames (optimizado)"""
        batch_size, frame_length = frames_batch.shape
        energies_batch = np.zeros((batch_size, self.n_bands), dtype=np.float64)
        
        for i in range(batch_size):
            frame = frames_batch[i].copy()  # Copia escribible
            wp = pywt.WaveletPacket(data=frame, wavelet=self.wavelet, 
                                  mode='symmetric', maxlevel=self.level)
            nodes = [n.path for n in wp.get_level(self.level, order='freq')]
            
            for j, node_path in enumerate(nodes):
                node_data = wp[node_path].data
                energies_batch[i, j] = np.sum(node_data.astype(np.float64) ** 2)
        
        return energies_batch
    
    def aggregate_to_88_batch(self, energies_batch):
        """Agrega energías a los 88 bins de teclas (vectorizado)"""
        return energies_batch @ self._overlap_matrix

def note_name(midi_note):
    """Conversión optimizada de MIDI a nombre de nota"""
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[midi_note % 12]}{(midi_note // 12) - 1}"

def piano_key_centers():
    """Calcula los centros de frecuencia de las teclas del piano"""
    mids = np.arange(MIDI_MIN, MIDI_MAX + 1, dtype=np.int32)
    freqs = 440.0 * (2.0 ** ((mids - 69) / 12.0))
    return mids, freqs

def piano_key_edges():
    """Calcula los bordes de frecuencia de las teclas del piano"""
    mids, centers = piano_key_centers()
    edges = np.sqrt(centers[:-1] * centers[1:])
    low = centers[0] / np.sqrt(2 ** (1/12))
    high = centers[-1] * np.sqrt(2 ** (1/12))
    return np.concatenate([[low], edges, [high]])

# Precalcular constantes globales
KEY_EDGES = piano_key_edges()
MIDI_CENTERS, FREQ_CENTERS = piano_key_centers()

def frame_audio_optimized(x, sr, frame_sec, hop_sec):
    """Frameado optimizado que evita problemas de memoria de solo lectura"""
    frame_len = int(round(frame_sec * sr))
    hop_len = int(round(hop_sec * sr))
    
    if len(x) < frame_len:
        return np.empty((0, frame_len), dtype=np.float32), frame_len, hop_len
    
    n_frames = 1 + (len(x) - frame_len) // hop_len
    
    # Crear frames como copias explícitas (evita problemas de solo lectura)
    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        frames[i] = x[start:end]
    
    return frames, frame_len, hop_len

def resample_audio(x, orig_sr, target_sr):
    """Resampleo de audio optimizado usando scipy"""
    if orig_sr == target_sr:
        return x
    
    duration = len(x) / orig_sr
    target_samples = int(duration * target_sr)
    
    return signal.resample(x, target_samples).astype(np.float32)

def extract_features_wpt_optimized(wav_path, meta, batch_size=32):
    """Extracción de características optimizada con procesamiento por lotes"""
    fs = meta.get("fs", DEFAULTS["fs"])
    frame_sec = meta.get("frame_sec", DEFAULTS["frame_sec"])
    hop_sec = meta.get("hop_sec", DEFAULTS["hop_sec"])
    wavelet = meta.get("wavelet", DEFAULTS["wavelet"])
    level = meta.get("level", DEFAULTS["level"])

    # Cargar audio
    x, sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
    if x.ndim > 1: 
        x = x.mean(axis=1)  # Mono

    # Resampleo optimizado
    if sr != fs:
        x = resample_audio(x, sr, fs)
        sr = fs

    # Frameado
    frames, N, H = frame_audio_optimized(x, sr, frame_sec, hop_sec)
    if frames.shape[0] == 0:
        return np.empty((0, 88), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # Cálculo RMS vectorizado
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    env_db = 20 * np.log10(rms)
    peak_db = np.max(env_db)

    # Procesamiento WPT por lotes
    wpt_processor = WPTProcessor(wavelet=wavelet, level=level, fs=fs)
    n_frames = frames.shape[0]
    feats = []
    
    for start_idx in range(0, n_frames, batch_size):
        end_idx = min(start_idx + batch_size, n_frames)
        batch_frames = frames[start_idx:end_idx]
        
        # Procesar lote
        energies_batch = wpt_processor.compute_band_energies_batch(batch_frames)
        feat_batch = wpt_processor.aggregate_to_88_batch(energies_batch)
        feat_batch = np.log10(feat_batch + 1e-12).astype(np.float32)
        
        feats.append(feat_batch)
    
    X = np.vstack(feats) if feats else np.empty((0, 88), dtype=np.float32)
    return X, env_db - peak_db  # dB relativos al pico

def median_filter_optimized(arr, kernel_size):
    """Filtro de mediana optimizado usando scipy"""
    if kernel_size <= 1 or arr.size == 0:
        return arr
    
    # Asegurar que el kernel size sea impar
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    return median_filter(arr, size=kernel_size, mode='nearest')

def frames_to_events_optimized(labels, hop_sec, frame_sec, idx_to_midi, min_frames=2):
    """Detección de eventos optimizada usando operaciones vectorizadas"""
    if len(labels) == 0:
        return []
    
    # Encontrar cambios en las etiquetas
    change_points = np.where(np.diff(labels) != 0)[0] + 1
    starts = np.concatenate([[0], change_points])
    ends = np.concatenate([change_points, [len(labels)]])
    
    events = []
    for start, end in zip(starts, ends):
        label = labels[start]
        if label < 0 or (end - start) < min_frames:
            continue
            
        midi = idx_to_midi[label]
        start_s = start * hop_sec
        end_s = (end - 1) * hop_sec + frame_sec
        duration_s = max(0.0, end_s - start_s)
        
        events.append({
            "midi": int(midi),
            "name": note_name(int(midi)),
            "start_s": float(start_s),
            "end_s": float(end_s),
            "dur_s": float(duration_s),
            "frames": int(end - start)
        })
    
    return events

def pick_model_optimized(models_dir: Path, model_path: Path = None):
    """Selección de modelo optimizada con fallbacks"""
    if model_path and model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f), model_path.name
    
    # Orden de preferencia de modelos
    model_preference = [
        "rf_best.pkl", "linear_svc.pkl", "logreg_saga.pkl", 
        "rf.pkl", "logreg.pkl", "knn_best.pkl", "knn5.pkl"
    ]
    
    for model_name in model_preference:
        model_path = models_dir / model_name
        if model_path.exists():
            with open(model_path, "rb") as f:
                return pickle.load(f), model_path.name
    
    raise FileNotFoundError(f"No se encontraron modelos en {models_dir}")

def analyze_events(events):
    """Análisis optimizado de eventos detectados"""
    if not events:
        return None, "No se detectaron eventos"
    
    # Calcular duración por nota (vectorizado)
    midi_notes = np.array([e["midi"] for e in events])
    durations = np.array([e["dur_s"] for e in events])
    
    # Nota dominante por duración total
    unique_notes = np.unique(midi_notes)
    total_durations = np.array([durations[midi_notes == note].sum() for note in unique_notes])
    dominant_idx = np.argmax(total_durations)
    dominant_midi = unique_notes[dominant_idx]
    
    return dominant_midi, {
        "total_events": len(events),
        "unique_notes": len(unique_notes),
        "total_duration": durations.sum(),
        "avg_duration": durations.mean()
    }

def main():
    parser = argparse.ArgumentParser(
        description="Inferencia optimizada: WAV → nota(s) usando pipeline WPT + modelo entrenado"
    )
    parser.add_argument("--wav", type=Path, required=True, help="Archivo WAV de entrada")
    parser.add_argument("--feat_dir", type=Path, default=Path("features_wpt_optimized"), 
                       help="Directorio con características y scaler")
    parser.add_argument("--model", type=Path, default=None, 
                       help="Ruta específica al modelo .pkl (opcional)")
    parser.add_argument("--models_dir", type=Path, default=Path("models"),
                       help="Directorio con modelos entrenados")
    parser.add_argument("--median_k", type=int, default=7, 
                       help="Tamaño del kernel para filtro de mediana")
    parser.add_argument("--silence_db", type=float, default=50.0, 
                       help="Umbral de silencio en dB relativos")
    parser.add_argument("--min_frames_event", type=int, default=2, 
                       help="Mínimo número de frames para considerar evento")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Tamaño de lote para procesamiento WPT")
    parser.add_argument("--midi_out", type=Path, default=None, 
                       help="Archivo MIDI de salida")
    parser.add_argument("--csv_out", type=Path, default=None, 
                       help="Archivo CSV de salida")
    parser.add_argument("--json_out", type=Path, default=None,
                       help="Archivo JSON con resultados detallados")
    
    args = parser.parse_args()

    print(f"🎵 Procesando: {args.wav.name}")
    
    # Cargar configuración
    meta_path = args.feat_dir / "meta.json"
    scaler_path = args.feat_dir / "scaler.pkl"
    
    if not meta_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"No se encontraron meta.json o scaler.pkl en {args.feat_dir}")
    
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    idx_to_midi = np.array(meta.get("keys_midi", list(range(MIDI_MIN, MIDI_MAX + 1))), dtype=np.int32)
    hop_sec = meta.get("hop_sec", DEFAULTS["hop_sec"])
    frame_sec = meta.get("frame_sec", DEFAULTS["frame_sec"])

    # Cargar modelo
    try:
        model, model_name = pick_model_optimized(args.models_dir, args.model)
        print(f"🤖 Modelo: {model_name}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # Extraer características
    print("🔨 Extrayendo características...")
    X_raw, rel_env_db = extract_features_wpt_optimized(
        args.wav, meta, batch_size=args.batch_size
    )
    
    if X_raw.shape[0] == 0:
        print("❌ Audio demasiado corto para procesar")
        return

    # Preprocesar y predecir
    X = scaler.transform(X_raw).astype(np.float32)
    y_pred = model.predict(X).astype(np.int16)

    # Filtrar silencio
    if args.silence_db is not None:
        silence_mask = rel_env_db < -abs(args.silence_db)
        y_pred[silence_mask] = -1
        print(f"🔇 Frames de silencio: {np.sum(silence_mask)}/{len(y_pred)}")

    # Suavizar predicciones
    if args.median_k > 1:
        valid_mask = y_pred >= 0
        if np.any(valid_mask):
            y_pred[valid_mask] = median_filter_optimized(
                y_pred[valid_mask], args.median_k
            )
        print("✅ Predicciones suavizadas")

    # Detectar eventos
    events = frames_to_events_optimized(
        y_pred, hop_sec, frame_sec, idx_to_midi, args.min_frames_event
    )

    # Analizar resultados
    dominant_midi, stats = analyze_events(events)
    
    if dominant_midi is None:
        print("🎵 No se detectaron notas (posible silencio o umbral muy alto)")
        return

    # Mostrar resultados
    print(f"\n🎯 NOTA DOMINANTE: {note_name(dominant_midi)} (MIDI {dominant_midi})")
    print(f"📊 Estadísticas: {stats['total_events']} eventos, "
          f"{stats['unique_notes']} notas únicas, "
          f"duración total: {stats['total_duration']:.2f}s")
    
    print("\n📝 Primeros 5 eventos:")
    for e in events[:5]:
        print(f"  {e['name']:>4s}  {e['start_s']:6.3f}s → {e['end_s']:6.3f}s  "
              f"({e['dur_s']:5.3f}s)")

    # Guardar resultados
    output_files = []
    
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=events[0].keys())
            writer.writeheader()
            writer.writerows(events)
        output_files.append(f"CSV: {args.csv_out}")

    if args.midi_out:
        args.midi_out.parent.mkdir(parents=True, exist_ok=True)
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)  # Piano acústico
        
        for e in events:
            inst.notes.append(pretty_midi.Note(
                velocity=90,
                pitch=int(e["midi"]),
                start=float(e["start_s"]),
                end=float(e["end_s"])
            ))
        
        pm.instruments.append(inst)
        pm.write(str(args.midi_out))
        output_files.append(f"MIDI: {args.midi_out}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "input_file": str(args.wav),
            "model_used": model_name,
            "dominant_note": {
                "midi": int(dominant_midi),
                "name": note_name(dominant_midi)
            },
            "statistics": stats,
            "events": events,
            "processing_parameters": {
                "silence_db": args.silence_db,
                "median_kernel": args.median_k,
                "min_frames_event": args.min_frames_event,
                "batch_size": args.batch_size
            }
        }
        
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        output_files.append(f"JSON: {args.json_out}")

    if output_files:
        print(f"\n💾 Archivos guardados:")
        for file in output_files:
            print(f"   {file}")

if __name__ == "__main__":
    main()