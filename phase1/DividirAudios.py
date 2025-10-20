# make_splits.py
import os
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict
import pandas as pd

DEFAULT_META = Path("metadata/index.csv")
DEFAULT_OUT  = Path("splits")
DEFAULT_LINK = Path("split_audio")

def make_link(src: Path, dst: Path, mode: str = "symlink"):
    """
    Crea un enlace de 'dst' -> 'src' en el modo indicado.
    - symlink : enlace simbólico (requiere permisos en Windows)
    - hardlink: enlace duro (mismo volumen/unidad)
    - copy    : copia el archivo
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return

    if mode == "symlink":
        try:
            os.symlink(src, dst)  # src debe ser absoluto para evitar roturas
            return
        except Exception as e:
            raise RuntimeError(f"symlink falló: {e}")
    elif mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception as e:
            raise RuntimeError(f"hardlink falló: {e}")
    elif mode == "copy":
        try:
            shutil.copy2(src, dst)
            return
        except Exception as e:
            raise RuntimeError(f"copy falló: {e}")
    else:
        raise ValueError(f"Modo inválido: {mode}")

def make_key(row, cols):
    return tuple(row[c] for c in cols)

def stratified_split(df: pd.DataFrame, strata_cols, train=0.70, valid=0.15, test=0.15, seed=123):
    assert abs(train + valid + test - 1.0) < 1e-6
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for i, r in df.iterrows():
        buckets[make_key(r, strata_cols)].append(i)

    tr, va, te = [], [], []
    for _, idxs in buckets.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = int(round(n * train))
        n_va = int(round(n * valid))
        n_te = n - n_tr - n_va
        tr += idxs[:n_tr]
        va += idxs[n_tr:n_tr+n_va]
        te += idxs[n_tr+n_va:]
    return df.loc[tr].copy(), df.loc[va].copy(), df.loc[te].copy()

def mirror_links(split_df: pd.DataFrame, root_link: Path, split_name: str, mode: str = "symlink"):
    """
    Crea estructura: split_audio/<split>/<note>/<archivo.wav>
    Enlaza SOLO .wav que existan.
    Usa rutas ABSOLUTAS como origen para evitar symlinks rotos.
    """
    errors = 0
    for _, r in split_df.iterrows():
        wav_rel = Path(str(r["filepath"]))
        if wav_rel.suffix.lower() != ".wav":
            continue
        src = wav_rel.resolve()  # <-- ABSOLUTO!
        if not src.exists():
            errors += 1
            continue
        note = str(r.get("note", "unknown"))
        dst = root_link / split_name / note / src.name
        try:
            make_link(src, dst, mode=mode)
        except Exception as e:
            errors += 1
            print(f"⚠️  No pude enlazar {src} → {dst} ({e})")
    if errors:
        print(f"⚠️  Aviso: {errors} archivos no se enlazaron (ver mensajes arriba).")

def main():
    ap = argparse.ArgumentParser(description="Crear splits estratificados (train/valid/test) para audios de piano.")
    ap.add_argument("--meta_csv", type=Path, default=DEFAULT_META)
    ap.add_argument("--out_dir",  type=Path, default=DEFAULT_OUT)
    ap.add_argument("--link_dir", type=Path, default=DEFAULT_LINK)
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--valid", type=float, default=0.15)
    ap.add_argument("--test",  type=float, default=0.15)
    ap.add_argument("--seed",  type=int,   default=123)
    ap.add_argument("--no_links", action="store_true", help="No crear split_audio/*")
    ap.add_argument("--strata", type=str, default="note",
                    help="Columnas de estratificación separadas por coma. Ej.: 'note' o 'note,velocity,articulation,pedal'")
    ap.add_argument("--mode", type=str, default="symlink",
                    choices=["symlink","hardlink","copy"],
                    help="Cómo materializar split_audio: symlink (default), hardlink (Windows-friendly, mismo disco), copy (seguro)")
    args = ap.parse_args()

    # Cargar CSV
    if not args.meta_csv.exists():
        raise SystemExit(f"ERROR: no existe {args.meta_csv}")
    df = pd.read_csv(args.meta_csv)

    # Columnas mínimas
    if "filepath" not in df.columns:
        raise SystemExit("ERROR: falta columna 'filepath' en metadata/index.csv")
    for col in ["note", "midi", "velocity", "articulation", "pedal"]:
        if col not in df.columns:
            df[col] = None

    # Filtrar archivos existentes
    df["exists"] = df["filepath"].apply(lambda p: Path(p).exists())
    miss = int((~df["exists"]).sum())
    if miss:
        print(f"⚠️  {miss} rutas del CSV no existen; se ignorarán.")
        df = df[df["exists"]].copy()
    df.drop(columns=["exists"], inplace=True)

    # Columnas de estratificación
    strata_cols = [c.strip() for c in args.strata.split(",") if c.strip()]
    for c in strata_cols:
        if c not in df.columns:
            raise SystemExit(f"ERROR: columna de estratificación '{c}' no existe en el CSV.")
    print(f"Estratificación por: {strata_cols}")

    # Split
    train_df, valid_df, test_df = stratified_split(
        df, strata_cols=strata_cols,
        train=args.train, valid=args.valid, test=args.test, seed=args.seed
    )

    # Guardar CSVs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train.csv").write_text(train_df.to_csv(index=False), encoding="utf-8")
    (args.out_dir / "valid.csv").write_text(valid_df.to_csv(index=False), encoding="utf-8")
    (args.out_dir / "test.csv").write_text(test_df.to_csv(index=False),  encoding="utf-8")
    print(f"✅ CSVs en {args.out_dir.resolve()}")
    print(f"   train: {len(train_df)} | valid: {len(valid_df)} | test: {len(test_df)}")

    # Estructura de audios (opcional)
    if not args.no_links:
        print(f"Creando estructura de '{args.mode}' en {args.link_dir.resolve()} …")
        for name, split in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
            mirror_links(split, args.link_dir, name, mode=args.mode)
        print("🔗 Listo.")

if __name__ == "__main__":
    main()
