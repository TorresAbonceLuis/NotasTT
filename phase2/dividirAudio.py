# make_splits_optimized.py
import os
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

DEFAULT_META = Path("metadata/index.csv")  # Actualizado a tu nuevo CSV
DEFAULT_OUT  = Path("splits")
DEFAULT_LINK = Path("split_audio")

def make_link(src: Path, dst: Path, mode: str = "symlink"):
    """
    Crea enlaces de forma robusta con mejor manejo de errores
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Eliminar destino si ya existe
    if dst.exists() or dst.is_symlink():
        try:
            if dst.is_symlink():
                dst.unlink()
            else:
                dst.unlink()
        except:
            pass

    try:
        if mode == "symlink":
            # Usar ruta absoluta para symlinks más robustos
            src_absolute = src.resolve()
            dst.symlink_to(src_absolute)
        elif mode == "hardlink":
            os.link(src, dst)
        elif mode == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Modo inválido: {mode}")
        return True
    except (OSError, ValueError, shutil.SameFileError) as e:
        print(f"⚠️  Error creando {mode} {src} → {dst}: {e}")
        return False

def make_key(row, cols):
    """Crea clave de estratificación más robusta"""
    try:
        return tuple(str(row[c]) if pd.notna(row[c]) else "unknown" for c in cols)
    except KeyError as e:
        print(f"⚠️  Columna faltante en estratificación: {e}")
        return ("unknown",)

def stratified_split_optimized(df: pd.DataFrame, strata_cols, train=0.70, valid=0.15, test=0.15, seed=123, min_samples_per_stratum=1):
    """
    Split estratificado mejorado con validación de datos
    """
    # Validar proporciones
    total = train + valid + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Las proporciones deben sumar 1.0, no {total}")
    
    # Verificar que hay suficientes datos
    if len(df) < 10:
        print("⚠️  Dataset muy pequeño, usando split aleatorio simple")
        return simple_split(df, train, valid, test, seed)
    
    rng = random.Random(seed)
    buckets = defaultdict(list)
    
    # Agrupar por estratos
    for i, r in df.iterrows():
        key = make_key(r, strata_cols)
        buckets[key].append(i)
    
    print(f"📊 Estratificación: {len(buckets)} grupos únicos")
    
    # Estadísticas de grupos
    group_sizes = [len(indices) for indices in buckets.values()]
    print(f"📈 Tamaño de grupos - Min: {min(group_sizes)}, Max: {max(group_sizes)}, Avg: {np.mean(group_sizes):.1f}")
    
    tr, va, te = [], [], []
    
    for stratum_key, indices in buckets.items():
        n = len(indices)
        
        if n < min_samples_per_stratum:
            print(f"⚠️  Estrato {stratum_key} tiene muy pocas muestras ({n}), asignando a train")
            tr.extend(indices)
            continue
            
        rng.shuffle(indices)
        
        # Calcular tamaños exactos
        n_tr = max(1, int(n * train))
        n_va = max(0, int(n * valid))
        n_te = n - n_tr - n_va
        
        # Ajustar si n_te es negativo
        if n_te < 0:
            n_te = 0
            n_va = max(0, n - n_tr)
        
        tr.extend(indices[:n_tr])
        va.extend(indices[n_tr:n_tr + n_va])
        te.extend(indices[n_tr + n_va:])
    
    # Verificar que tenemos datos en cada split
    if not tr:
        raise ValueError("No hay datos en training split")
    
    print(f"✅ Split final - Train: {len(tr)}, Valid: {len(va)}, Test: {len(te)}")
    
    return df.loc[tr].copy(), df.loc[va].copy(), df.loc[te].copy()

def simple_split(df: pd.DataFrame, train=0.70, valid=0.15, test=0.15, seed=123):
    """
    Split simple cuando hay muy pocos datos para estratificación
    """
    rng = random.Random(seed)
    indices = list(df.index)
    rng.shuffle(indices)
    
    n = len(indices)
    n_tr = int(n * train)
    n_va = int(n * valid)
    n_te = n - n_tr - n_va
    
    tr = indices[:n_tr]
    va = indices[n_tr:n_tr + n_va]
    te = indices[n_tr + n_va:]
    
    return df.loc[tr].copy(), df.loc[va].copy(), df.loc[te].copy()

def mirror_links_optimized(split_df: pd.DataFrame, root_link: Path, split_name: str, mode: str = "symlink"):
    """
    Crea estructura de enlaces con mejor reporte y manejo de errores
    """
    if split_df.empty:
        print(f"⚠️  No hay datos para {split_name}, saltando enlaces")
        return
    
    print(f"🔗 Creando enlaces para {split_name} ({len(split_df)} archivos)...")
    
    success_count = 0
    error_count = 0
    missing_files = []
    
    for _, r in split_df.iterrows():
        wav_path = Path(str(r["filepath"]))
        
        if not wav_path.exists():
            missing_files.append(str(wav_path))
            error_count += 1
            continue
            
        # Crear estructura de directorios organizada
        note = str(r.get("note", "unknown")).replace('#', 's')  # Sanitizar nombres
        dst = root_link / split_name / note / wav_path.name
        
        if make_link(wav_path, dst, mode):
            success_count += 1
        else:
            error_count += 1
    
    # Reporte detallado
    print(f"   ✅ Éxitos: {success_count}")
    if error_count > 0:
        print(f"   ❌ Errores: {error_count}")
    if missing_files:
        print(f"   📝 Archivos faltantes: {len(missing_files)}")
        for f in missing_files[:5]:  # Mostrar solo los primeros 5
            print(f"      - {f}")

def validate_dataframe(df):
    """
    Valida y limpia el DataFrame
    """
    # Verificar columnas requeridas
    required_cols = ["filepath", "midi"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en CSV: {missing_cols}")
    
    # Verificar que los archivos existen
    df["exists"] = df["filepath"].apply(lambda p: Path(p).exists())
    missing_count = (~df["exists"]).sum()
    
    if missing_count > 0:
        print(f"⚠️  {missing_count} archivos no existen, serán eliminados del dataset")
        missing_files = df[~df["exists"]]["filepath"].tolist()
        for f in missing_files[:5]:
            print(f"   - {f}")
        if len(missing_files) > 5:
            print(f"   ... y {len(missing_files) - 5} más")
        
        df = df[df["exists"]].copy()
    
    df = df.drop(columns=["exists"])
    
    # Verificar que tenemos datos
    if df.empty:
        raise ValueError("No hay datos válidos después de la limpieza")
    
    # Estadísticas del dataset
    print(f"📊 Dataset final: {len(df)} archivos")
    if "note" in df.columns:
        note_counts = df["note"].value_counts()
        print(f"🎵 Distribución de notas: {len(note_counts)} notas únicas")
        print(f"   Notas más comunes: {', '.join(note_counts.head(5).index.tolist())}")
    
    return df

def analyze_split_distribution(train_df, valid_df, test_df, strata_cols):
    """
    Analiza la distribución de los splits
    """
    print("\n📈 ANÁLISIS DE DISTRIBUCIÓN:")
    
    for col in strata_cols:
        if col not in train_df.columns:
            continue
            
        print(f"\n{col}:")
        for name, df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
            if not df.empty:
                value_counts = df[col].value_counts()
                print(f"  {name:6}: {len(value_counts)} valores únicos, "
                      f"distribución: {', '.join([f'{k}({v})' for k, v in value_counts.head(3).items()])}")

def main():
    ap = argparse.ArgumentParser(
        description="Crear splits estratificados optimizados para audios de piano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Split básico
  python make_splits_optimized.py
  
  # Split con diferentes proporciones
  python make_splits_optimized.py --train 0.8 --valid 0.1 --test 0.1
  
  # Split estratificando por múltiples columnas
  python make_splits_optimized.py --strata "note,velocity"
  
  # Usar copias en lugar de symlinks (más seguro en Windows)
  python make_splits_optimized.py --mode copy
        """
    )
    
    ap.add_argument("--meta_csv", type=Path, default=DEFAULT_META,
                   help="Archivo CSV de metadata (por defecto: metadata/index.csv)")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT,
                   help="Directorio de salida para los CSVs de splits")
    ap.add_argument("--link_dir", type=Path, default=DEFAULT_LINK,
                   help="Directorio para estructura de enlaces")
    ap.add_argument("--train", type=float, default=0.70,
                   help="Proporción para training (por defecto: 0.70)")
    ap.add_argument("--valid", type=float, default=0.15,
                   help="Proporción para validación (por defecto: 0.15)")
    ap.add_argument("--test", type=float, default=0.15,
                   help="Proporción para test (por defecto: 0.15)")
    ap.add_argument("--seed", type=int, default=42,
                   help="Semilla para reproducibilidad (por defecto: 42)")
    ap.add_argument("--no_links", action="store_true",
                   help="No crear estructura de enlaces")
    ap.add_argument("--strata", type=str, default="note",
                   help="Columnas para estratificación separadas por coma (por defecto: 'note')")
    ap.add_argument("--mode", type=str, default="symlink",
                   choices=["symlink", "hardlink", "copy"],
                   help="Tipo de enlace: symlink, hardlink, o copy (por defecto: symlink)")
    ap.add_argument("--min_samples", type=int, default=1,
                   help="Mínimo de muestras por estrato (por defecto: 1)")
    
    args = ap.parse_args()

    print("🚀 INICIANDO CREACIÓN DE SPLITS OPTIMIZADA")
    print("=" * 50)
    
    # Validar archivo de metadata
    if not args.meta_csv.exists():
        raise SystemExit(f"❌ ERROR: No existe el archivo {args.meta_csv}")
    
    # Cargar y validar datos
    print("📁 Cargando metadata...")
    try:
        df = pd.read_csv(args.meta_csv)
        print(f"   Archivo cargado: {len(df)} registros")
    except Exception as e:
        raise SystemExit(f"❌ ERROR leyendo {args.meta_csv}: {e}")
    
    df = validate_dataframe(df)
    
    # Preparar columnas de estratificación
    strata_cols = [c.strip() for c in args.strata.split(",") if c.strip()]
    missing_strata_cols = [c for c in strata_cols if c not in df.columns]
    
    if missing_strata_cols:
        print(f"⚠️  Columnas de estratificación faltantes: {missing_strata_cols}")
        print("   Usando estratificación simple...")
        strata_cols = [c for c in strata_cols if c in df.columns]
        if not strata_cols:
            strata_cols = ["note"] if "note" in df.columns else []
    
    print(f"🎯 Estratificando por: {strata_cols if strata_cols else 'Ninguna (split simple)'}")
    
    # Crear splits
    try:
        train_df, valid_df, test_df = stratified_split_optimized(
            df, 
            strata_cols=strata_cols,
            train=args.train, 
            valid=args.valid, 
            test=args.test, 
            seed=args.seed,
            min_samples_per_stratum=args.min_samples
        )
    except Exception as e:
        raise SystemExit(f"❌ ERROR en stratified split: {e}")
    
    # Analizar distribución
    analyze_split_distribution(train_df, valid_df, test_df, strata_cols)
    
    # Guardar CSVs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = args.out_dir / "train.csv"
    valid_csv = args.out_dir / "valid.csv" 
    test_csv = args.out_dir / "test.csv"
    
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"\n✅ SPLITS GUARDADOS EN {args.out_dir.resolve()}")
    print(f"   📄 train.csv:  {len(train_df)} muestras")
    print(f"   📄 valid.csv:  {len(valid_df)} muestras") 
    print(f"   📄 test.csv:   {len(test_df)} muestras")
    
    # Crear estructura de enlaces
    if not args.no_links:
        print(f"\n🔗 CREANDO ESTRUCTURA DE ENLACES ({args.mode})...")
        print(f"   Directorio: {args.link_dir.resolve()}")
        
        # Limpiar directorio existente si es necesario
        if args.link_dir.exists() and args.mode == "copy":
            print("   🧹 Limpiando directorio existente...")
            shutil.rmtree(args.link_dir)
        
        for name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
            mirror_links_optimized(split_df, args.link_dir, name, mode=args.mode)
        
        print("   ✅ Estructura de enlaces completada")
    
    print("\n🎉 PROCESO COMPLETADO EXITOSAMENTE!")
    print(f"   Next: Ejecuta la extracción de características con estos splits")

if __name__ == "__main__":
    main()