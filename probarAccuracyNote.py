# analyze_confusion.py
import argparse, json, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

MIDI_MIN, MIDI_MAX = 21, 108

def note_name(m):
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[m%12]}{(m//12)-1}"

def load_features(feat_dir: Path):
    X_train = np.load(feat_dir / "X_train.npy"); y_train = np.load(feat_dir / "y_train.npy")
    X_valid = np.load(feat_dir / "X_valid.npy"); y_valid = np.load(feat_dir / "y_valid.npy")
    X_test  = np.load(feat_dir / "X_test.npy");  y_test  = np.load(feat_dir / "y_test.npy")
    with open(feat_dir / "meta.json","r",encoding="utf-8") as f:
        meta = json.load(f)
    idx_to_midi = np.array(meta.get("keys_midi", list(range(MIDI_MIN, MIDI_MAX+1))), dtype=np.int32)
    names = [note_name(int(m)) for m in idx_to_midi]
    return (X_train,y_train),(X_valid,y_valid),(X_test,y_test), idx_to_midi, names

def pick_model(models_dir: Path, model_path: Path=None):
    if model_path is not None:
        with open(model_path, "rb") as f: return pickle.load(f), model_path.name
    # heurística: preferir rf_best, luego linear_svc, luego logreg_saga, luego knn_best
    candidates = ["rf_best.pkl","linear_svc.pkl","logreg_saga.pkl","knn_best.pkl","logreg.pkl","knn5.pkl","rf.pkl"]
    for c in candidates:
        p = models_dir / c
        if p.exists():
            with open(p,"rb") as f: return pickle.load(f), p.name
    raise SystemExit(f"No encontré modelo en {models_dir}. Pasa --model.")

def plot_cm(cm, labels, title, out_png, normalize=False):
    if normalize:
        with np.errstate(invalid='ignore'):
            cm = cm.astype(np.float64)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums!=0)
    plt.figure(figsize=(12,10))
    im = plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_step = max(1, len(labels)//22)  # no saturar ticks
    ticks = np.arange(0, len(labels), tick_step)
    plt.xticks(ticks, [labels[i] for i in ticks], rotation=90)
    plt.yticks(ticks, [labels[i] for i in ticks])
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

def per_class_accuracy(cm):
    tp = np.diag(cm).astype(np.float64)
    totals = cm.sum(axis=1).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        acc = np.divide(tp, totals, out=np.zeros_like(tp), where=totals!=0)
    return acc

def top_confusions(cm, labels, k=15):
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    pairs = []
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            if cm2[i,j] > 0:
                pairs.append((int(cm2[i,j]), labels[i], labels[j], i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return pairs[:k]

def main():
    ap = argparse.ArgumentParser(description="Genera matriz de confusión y accuracy por nota.")
    ap.add_argument("--feat_dir", type=Path, default=Path("features_wpt"))
    ap.add_argument("--models_dir", type=Path, default=Path("models"))
    ap.add_argument("--model", type=Path, default=None, help="Ruta directa a un .pkl si quieres forzar modelo")
    ap.add_argument("--out_dir", type=Path, default=Path("analysis"))
    ap.add_argument("--split", type=str, default="test", choices=["valid","test"], help="Split a evaluar")
    ap.add_argument("--median_k", type=int, default=0, help="(opcional) suavizado de mediana sobre predicciones")
    args = ap.parse_args()

    (Xtr,ytr),(Xv,yv),(Xt,yt), idx_to_midi, labels = load_features(args.feat_dir)
    model, model_name = pick_model(args.models_dir, args.model)

    if args.split == "valid":
        X, y = Xv, yv
    else:
        X, y = Xt, yt

    y_pred = model.predict(X)

    # Suavizado opcional (suaviza etiquetas vecinas en la secuencia global; útil si X proviene de concatenar frames)
    if args.median_k and args.median_k > 1:
        k = args.median_k if args.median_k % 2 == 1 else args.median_k+1
        from scipy.ndimage import median_filter
        y_pred = median_filter(y_pred, size=k, mode="nearest")

    acc = accuracy_score(y, y_pred)
    print(f"Evaluando {args.split} con {model_name} → accuracy global: {acc:.4f}")

    # Matriz de confusión
    n_classes = len(idx_to_midi)
    cm = confusion_matrix(y, y_pred, labels=np.arange(n_classes, dtype=int))

    # Guardar PNGs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_cm(cm, labels, f"CM {args.split} - {model_name} (abs)", args.out_dir / f"cm_{args.split}_{model_name}_abs.png", normalize=False)
    plot_cm(cm, labels, f"CM {args.split} - {model_name} (fila %)", args.out_dir / f"cm_{args.split}_{model_name}_row.png", normalize=True)
    print("✅ Guardé:", args.out_dir / f"cm_{args.split}_{model_name}_abs.png")
    print("✅ Guardé:", args.out_dir / f"cm_{args.split}_{model_name}_row.png")

    # Accuracy por clase (nota) → CSV
    acc_per_note = per_class_accuracy(cm)
    rows = []
    for i,(m,lab,a) in enumerate(zip(idx_to_midi, labels, acc_per_note)):
        rows.append((i,int(m),lab,float(a)))
    import csv
    acc_csv = args.out_dir / f"accuracy_by_note_{args.split}_{model_name}.csv"
    with open(acc_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_index","midi","note","accuracy"])
        w.writerows(rows)
    print("📄 CSV accuracy por nota:", acc_csv)

    # Top confusiones
    pairs = top_confusions(cm, labels, k=20)
    top_csv = args.out_dir / f"top_confusions_{args.split}_{model_name}.csv"
    with open(top_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["count","true_note","pred_note","true_idx","pred_idx"])
        for c, t, p, ti, pi in pairs:
            w.writerow([c, t, p, ti, pi])
    print("📄 CSV top confusiones:", top_csv)

if __name__ == "__main__":
    main()
