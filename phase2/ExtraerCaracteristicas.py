import argparse, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import pywt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

FS_DEFAULT = 44100
FRAME_SEC = 0.050
HOP_SEC   = 0.025
WAVELET   = "db8"
LEVEL     = 9
MIDI_MIN, MIDI_MAX = 21, 108  # A0..C8

def note_name(m):
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[m%12]}{(m//12)-1}"

def piano_key_centers():
    mids = np.arange(MIDI_MIN, MIDI_MAX+1)
    freqs = 440.0 * (2.0 ** ((mids - 69) / 12.0))
    return mids, freqs

def piano_key_edges():
    mids, centers = piano_key_centers()
    edges = np.sqrt(centers[:-1] * centers[1:])
    low  = centers[0] / np.sqrt(2**(1/12))
    high = centers[-1] * np.sqrt(2**(1/12))
    return np.concatenate([[low], edges, [high]])

KEY_EDGES = piano_key_edges()

class WPTProcessor:
    """Procesador optimizado de Wavelet Packet Transform"""
    
    def __init__(self, wavelet=WAVELET, level=LEVEL, fs=FS_DEFAULT):
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
        """Calcula energías de bandas WPT para un lote de frames"""
        batch_size, frame_length = frames_batch.shape
        energies_batch = np.zeros((batch_size, self.n_bands), dtype=np.float64)
        
        for i in range(batch_size):
            # Asegurar que el frame sea escribible haciendo una copia
            frame = frames_batch[i].copy()
            wp = pywt.WaveletPacket(data=frame, wavelet=self.wavelet, 
                                  mode='symmetric', maxlevel=self.level)
            nodes = [n.path for n in wp.get_level(self.level, order='freq')]
            
            energies = np.zeros(self.n_bands, dtype=np.float64)
            for j, node_path in enumerate(nodes):
                node_data = wp[node_path].data
                energies[j] = np.sum(node_data.astype(np.float64) ** 2)
            
            energies_batch[i] = energies
        
        return energies_batch
    
    def aggregate_to_88_batch(self, energies_batch):
        """Agrega energías a los 88 bins de teclas (vectorizado)"""
        # energies_batch: (batch_size, n_bands)
        # _overlap_matrix: (n_bands, 88)
        # resultado: (batch_size, 88)
        return energies_batch @ self._overlap_matrix

def frame_audio_optimized(x, sr, frame_sec=FRAME_SEC, hop_sec=HOP_SEC):
    """Frameado optimizado que asegura arrays escribibles"""
    frame_len = int(round(frame_sec * sr))
    hop_len = int(round(hop_sec * sr))
    
    if len(x) < frame_len:
        return np.empty((0, frame_len), dtype=np.float32)
    
    # Número de frames
    n_frames = 1 + (len(x) - frame_len) // hop_len
    
    # Crear array de frames directamente (no usar strided para evitar problemas de solo lectura)
    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        end = start + frame_len
        frames[i] = x[start:end]
    
    return frames

def process_file_optimized(wav_path, midi_label, wpt_processor, fs_expected=FS_DEFAULT, batch_size=32):
    """Procesamiento optimizado por lotes"""
    x, sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
    if x.ndim > 1: 
        x = x.mean(axis=1)
    if sr != fs_expected:
        raise ValueError(f"SR {sr} != esperado {fs_expected} en {wav_path}")
    
    frames = frame_audio_optimized(x, sr)
    if frames.shape[0] == 0:
        return np.empty((0, 88), dtype=np.float32), np.empty((0,), dtype=np.int64)
    
    n_frames = frames.shape[0]
    feats = []
    
    # Procesar por lotes
    for start_idx in range(0, n_frames, batch_size):
        end_idx = min(start_idx + batch_size, n_frames)
        batch_frames = frames[start_idx:end_idx]
        
        # Calcular energías por lote
        energies_batch = wpt_processor.compute_band_energies_batch(batch_frames)
        
        # Agregar a 88 bins (vectorizado)
        feat_batch = wpt_processor.aggregate_to_88_batch(energies_batch)
        
        # Compresión logarítmica
        feat_batch = np.log10(feat_batch + 1e-12).astype(np.float32)
        feats.append(feat_batch)
    
    X = np.vstack(feats) if feats else np.empty((0, 88), dtype=np.float32)
    y = np.full((X.shape[0],), int(midi_label - MIDI_MIN), dtype=np.int64)
    
    return X, y

def load_split(csv_path: Path):
    """Carga y filtra el dataset"""
    df = pd.read_csv(csv_path)
    # Verificar que los archivos existan
    valid_files = []
    for _, row in df.iterrows():
        file_path = Path(row["filepath"])
        if file_path.suffix.lower() == ".wav" and file_path.exists():
            valid_files.append(True)
        else:
            print(f"⚠️  Archivo no encontrado o no es WAV: {file_path}")
            valid_files.append(False)
    
    df = df[valid_files].copy()
    assert "midi" in df.columns and "filepath" in df.columns
    return df

def build_features_optimized(df, wpt_processor, scaler=None, fit_scaler=False, batch_size=32):
    """Construcción de características optimizada"""
    X_all, y_all = [], []
    
    for idx, (_, r) in enumerate(df.iterrows()):
        print(f"📁 Procesando archivo {idx+1}/{len(df)}: {Path(r['filepath']).name}")
        try:
            X, y = process_file_optimized(
                Path(r["filepath"]), 
                int(r["midi"]), 
                wpt_processor, 
                wpt_processor.fs,
                batch_size
            )
            if X.size == 0: 
                print(f"  ⚠️  Archivo muy corto: {Path(r['filepath']).name}")
                continue
            X_all.append(X)
            y_all.append(y)
        except Exception as e:
            print(f"  ❌ Error procesando {Path(r['filepath']).name}: {e}")
            continue
    
    if not X_all:
        return np.empty((0, 88), dtype=np.float32), np.empty((0,), dtype=np.int64), scaler
    
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(X_all)
    
    if scaler is not None:
        X_all = scaler.transform(X_all).astype(np.float32)
    
    return X_all, y_all, scaler

def main():
    global WAVELET, LEVEL, FRAME_SEC, HOP_SEC
    
    ap = argparse.ArgumentParser(description="Extracción WPT → 88 bins (A0..C8) - OPTIMIZADO")
    ap.add_argument("--splits_dir", type=Path, default=Path("splits"))
    ap.add_argument("--out_dir", type=Path, default=Path("features"))
    ap.add_argument("--fs", type=int, default=FS_DEFAULT)
    ap.add_argument("--wavelet", type=str, default=WAVELET)
    ap.add_argument("--level", type=int, default=LEVEL)
    ap.add_argument("--frame", type=float, default=FRAME_SEC)
    ap.add_argument("--hop", type=float, default=HOP_SEC)
    ap.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote para procesamiento")
    args = ap.parse_args()
    
    WAVELET = args.wavelet
    LEVEL = args.level
    FRAME_SEC = args.frame
    HOP_SEC = args.hop

    # Inicializar procesador WPT optimizado
    print("🔄 Inicializando procesador WPT optimizado...")
    wpt_processor = WPTProcessor(wavelet=args.wavelet, level=args.level, fs=args.fs)
    
    # Cargar datos
    print("📁 Cargando splits...")
    train_df = load_split(args.splits_dir / "train.csv")
    valid_df = load_split(args.splits_dir / "valid.csv")
    test_df = load_split(args.splits_dir / "test.csv")
    
    print(f"📊 Datasets: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

    # Procesar entrenamiento
    print("🔨 Procesando conjunto de entrenamiento...")
    X_train, y_train, scaler = build_features_optimized(
        train_df, wpt_processor, scaler=None, fit_scaler=True, batch_size=args.batch_size
    )
    
    # Crear directorio de salida y guardar
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "X_train.npy", X_train)
    np.save(args.out_dir / "y_train.npy", y_train)
    
    with open(args.out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Procesar validación y test
    print("🔨 Procesando conjunto de validación...")
    X_valid, y_valid, _ = build_features_optimized(
        valid_df, wpt_processor, scaler=scaler, fit_scaler=False, batch_size=args.batch_size
    )
    
    print("🔨 Procesando conjunto de test...")
    X_test, y_test, _ = build_features_optimized(
        test_df, wpt_processor, scaler=scaler, fit_scaler=False, batch_size=args.batch_size
    )
    
    # Guardar resultados
    np.save(args.out_dir / "X_valid.npy", X_valid)
    np.save(args.out_dir / "y_valid.npy", y_valid)
    np.save(args.out_dir / "X_test.npy", X_test)
    np.save(args.out_dir / "y_test.npy", y_test)

    # Guardar metadatos
    mids, _ = piano_key_centers()
    meta = {
        "fs": args.fs,
        "frame_sec": FRAME_SEC,
        "hop_sec": HOP_SEC,
        "wavelet": WAVELET,
        "level": LEVEL,
        "batch_size": args.batch_size,
        "keys_midi": mids.tolist(),
        "keys_name": [note_name(m) for m in mids],
        "key_edges_hz": KEY_EDGES.tolist(),
        "X_shapes": {
            "train": list(X_train.shape),
            "valid": list(X_valid.shape),
            "test": list(X_test.shape)
        }
    }
    
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Features optimizadas listas en", args.out_dir.resolve())
    print("   X_train:", X_train.shape, "X_valid:", X_valid.shape, "X_test:", X_test.shape)
    print(f"   Batch size utilizado: {args.batch_size}")

if __name__ == "__main__":
    main()