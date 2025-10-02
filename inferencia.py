# infer_note.py
import argparse, json, pickle
from pathlib import Path
import numpy as np
import soundfile as sf
import pywt
import pretty_midi

MIDI_MIN, MIDI_MAX = 21, 108
DEFAULTS = dict(fs=44100, frame_sec=0.050, hop_sec=0.025, wavelet="db8", level=9)

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

def frame_audio(x, sr, frame_sec, hop_sec):
    N = int(round(frame_sec * sr))
    H = int(round(hop_sec * sr))
    if len(x) < N:
        return np.empty((0, N), dtype=np.float32), N, H
    n = 1 + (len(x) - N) // H
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, N), strides=(x.strides[0]*H, x.strides[0]), writeable=False
    )
    return frames.copy(), N, H

def wpt_band_energies(x, fs, wavelet, level):
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode='symmetric', maxlevel=level)
    nodes = [n.path for n in wp.get_level(level, order='freq')]
    energies = np.array([float(np.sum(wp[n].data.astype(np.float64)**2)) for n in nodes], dtype=np.float64)
    B = 2 ** level
    bw = (fs / 2.0) / B
    idx = np.arange(B)
    f_low  = idx * bw
    f_high = (idx + 1) * bw
    return energies, f_low, f_high

def aggregate_to_88(energies, f_low, f_high):
    feat = np.zeros(88, dtype=np.float64)
    for b in range(len(energies)):
        bl, bh = f_low[b], f_high[b]
        if bh <= KEY_EDGES[0] or bl >= KEY_EDGES[-1]:
            continue
        i0 = np.searchsorted(KEY_EDGES, bl, side='right') - 1
        i1 = np.searchsorted(KEY_EDGES, bh, side='left')
        i0 = max(i0, 0); i1 = min(i1, 88)
        if i1 <= i0: continue
        for i in range(i0, i1):
            kl, kh = KEY_EDGES[i], KEY_EDGES[i+1]
            overlap = max(0.0, min(bh, kh) - max(bl, kl))
            if overlap > 0:
                feat[i] += energies[b] * (overlap / (bh - bl))
    return feat

def extract_features_wpt(wav_path, meta):
    fs = meta.get("fs", DEFAULTS["fs"])
    frame_sec = meta.get("frame_sec", DEFAULTS["frame_sec"])
    hop_sec   = meta.get("hop_sec", DEFAULTS["hop_sec"])
    wavelet   = meta.get("wavelet", DEFAULTS["wavelet"])
    level     = meta.get("level", DEFAULTS["level"])

    x, sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
    if x.ndim > 1: x = x.mean(axis=1)

    # Resample simple si el WAV no coincide con fs del meta (lineal)
    if sr != fs:
        t_src = np.linspace(0, len(x)/sr, num=len(x), endpoint=False)
        t_dst = np.linspace(0, len(x)/sr, num=int(round(len(x)*fs/sr)), endpoint=False)
        x = np.interp(t_dst, t_src, x).astype(np.float32)
        sr = fs

    frames, N, H = frame_audio(x, sr, frame_sec, hop_sec)
    if frames.shape[0] == 0:
        return np.empty((0,88), dtype=np.float32), np.empty((0,), dtype=np.float32)

    rms = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    env_db = 20*np.log10(rms); peak_db = float(np.max(env_db))
    feats = []
    for f in frames:
        e, fl, fh = wpt_band_energies(f, sr, wavelet, level)
        v = aggregate_to_88(e, fl, fh)
        v = np.log10(v + 1e-12).astype(np.float32)
        feats.append(v)
    X = np.vstack(feats)
    return X, env_db - peak_db  # dB relativos al pico

def median_filter_1d(arr, k):
    if k <= 1 or arr.size == 0: return arr
    k = k if k % 2 == 1 else k+1
    pad = k//2
    padded = np.pad(arr, (pad, pad), mode='edge')
    out = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = np.median(padded[i:i+k])
    return out.astype(arr.dtype)

def frames_to_events(labels, hop_sec, frame_sec, idx_to_midi, min_frames=2):
    events = []
    if len(labels) == 0: return events
    cur = labels[0]; start = 0
    for i in range(1, len(labels)):
        if labels[i] != cur:
            if cur >= 0:
                midi = idx_to_midi[cur]; end = i
                start_s = start*hop_sec; end_s = (end-1)*hop_sec + frame_sec
                events.append({"midi": int(midi), "name": note_name(int(midi)),
                               "start_s": float(start_s), "end_s": float(end_s),
                               "dur_s": float(max(0.0, end_s-start_s)),
                               "frames": int(end-start)})
            cur = labels[i]; start = i
    if cur >= 0:
        midi = idx_to_midi[cur]; end = len(labels)
        start_s = start*hop_sec; end_s = (end-1)*hop_sec + frame_sec
        events.append({"midi": int(midi), "name": note_name(int(midi)),
                       "start_s": float(start_s), "end_s": float(end_s),
                       "dur_s": float(max(0.0, end_s-start_s)),
                       "frames": int(end-start)})
    return [e for e in events if e["frames"] >= min_frames]

def pick_model(models_dir: Path, model_path: Path=None):
    if model_path is not None:
        with open(model_path, "rb") as f: return pickle.load(f), model_path.name
    for name in ["rf_best.pkl","linear_svc.pkl","logreg_saga.pkl","rf.pkl","logreg.pkl","knn_best.pkl","knn5.pkl"]:
        p = models_dir / name
        if p.exists():
            with open(p,"rb") as f: return pickle.load(f), p.name
    raise SystemExit(f"No encontré modelos en {models_dir}. Pasa --model.")

def main():
    ap = argparse.ArgumentParser(description="Inferencia: WAV → nota(s) usando el pipeline WPT + modelo entrenado")
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--feat_dir", type=Path, default=Path("features_wpt"))
    ap.add_argument("--model", type=Path, default=None, help="Ruta a .pkl (si no, elige el mejor disponible)")
    ap.add_argument("--median_k", type=int, default=7, help="Filtro de mediana sobre etiquetas por frame")
    ap.add_argument("--silence_db", type=float, default=50.0, help="Frames con RMS < pico-DB se marcan silencio")
    ap.add_argument("--min_frames_event", type=int, default=2)
    ap.add_argument("--midi_out", type=Path, default=None)
    ap.add_argument("--csv_out", type=Path, default=None)
    args = ap.parse_args()

    # Cargar meta y scaler
    meta = json.loads((args.feat_dir / "meta.json").read_text(encoding="utf-8"))
    with open(args.feat_dir / "scaler.pkl","rb") as f: scaler = pickle.load(f)
    idx_to_midi = np.array(meta.get("keys_midi", list(range(MIDI_MIN, MIDI_MAX+1))), dtype=np.int32)
    hop_sec   = meta.get("hop_sec", DEFAULTS["hop_sec"])
    frame_sec = meta.get("frame_sec", DEFAULTS["frame_sec"])

    # Modelo
    model, model_name = pick_model(Path("models"), args.model)
    print(f"Usando modelo: {model_name}")

    # Features del WAV
    X_raw, rel_env_db = extract_features_wpt(args.wav, meta)
    if X_raw.shape[0] == 0:
        print("WAV demasiado corto.")
        return
    X = scaler.transform(X_raw).astype(np.float32)

    # Predicción por frame
    y_pred = model.predict(X).astype(np.int16)

    # Silencio por energía (opcional)
    if args.silence_db is not None:
        y_pred[rel_env_db < -abs(args.silence_db)] = -1

    # Suavizado
    if args.median_k and args.median_k > 1:
        keep = (y_pred >= 0)
        y_tmp = y_pred.copy()
        y_tmp[keep] = median_filter_1d(y_pred[keep], args.median_k)
        y_pred = y_tmp

    # Eventos
    events = frames_to_events(y_pred, hop_sec, frame_sec, idx_to_midi, min_frames=args.min_frames_event)

    if not events:
        print("No se detectaron notas (posible silencio/umbral alto).")
        return

    # Nota dominante (por duración total)
    dur_by_midi = {}
    for e in events:
        dur_by_midi[e["midi"]] = dur_by_midi.get(e["midi"], 0.0) + e["dur_s"]
    dom_midi = max(dur_by_midi.items(), key=lambda kv: kv[1])[0]
    print(f"\n🎯 Nota dominante: {note_name(dom_midi)} (MIDI {dom_midi})")
    print(f"Eventos detectados: {len(events)}")
    for e in events[:5]:
        print(f"  {e['name']:>4s}  {e['start_s']:.3f}–{e['end_s']:.3f}s  ({e['dur_s']:.3f}s)")

    # Guardados opcionales
    if args.csv_out:
        import csv
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv_out,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["midi","name","start_s","end_s","dur_s","frames"])
            w.writeheader()
            for e in events: w.writerow(e)
        print("CSV:", args.csv_out)

    if args.midi_out:
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        for e in events:
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=int(e["midi"]),
                                               start=float(e["start_s"]), end=float(e["end_s"])))
        pm.instruments.append(inst)
        args.midi_out.parent.mkdir(parents=True, exist_ok=True)
        pm.write(str(args.midi_out))
        print("MIDI:", args.midi_out)

if __name__ == "__main__":
    main()
