import argparse, pickle, json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def load_features(root: Path):
    """Carga características y verifica el scaler"""
    X_train = np.load(root / "X_train.npy")
    X_valid = np.load(root / "X_valid.npy")
    X_test  = np.load(root / "X_test.npy")
    y_train = np.load(root / "y_train.npy")
    y_valid = np.load(root / "y_valid.npy")
    y_test  = np.load(root / "y_test.npy")
    
    # Verificar que existe el scaler
    scaler_path = root / "scaler.pkl"
    if scaler_path.exists():
        print("✅ Scaler encontrado - los datos ya están normalizados")
    else:
        print("⚠️  Advertencia: No se encontró scaler.pkl")
    
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def eval_model(name, model, Xv, yv, Xt, yt, cv_folds=5):
    """Evaluación más completa con validación cruzada"""
    # Validación cruzada en entrenamiento
    cv_scores = cross_val_score(model, Xv, yv, cv=min(cv_folds, len(np.unique(yv))), scoring='accuracy')
    
    # Predicciones
    yv_pred = model.predict(Xv)
    yt_pred = model.predict(Xt)
    
    # Métricas
    acc_v = accuracy_score(yv, yv_pred)
    acc_t = accuracy_score(yt, yt_pred)
    f1_v = f1_score(yv, yv_pred, average='weighted')
    f1_t = f1_score(yt, yt_pred, average='weighted')
    
    print(f"\n{name:20} | CV: {cv_scores.mean():.4f} (±{cv_scores.std():.4f}) | "
          f"Valid: {acc_v:.4f} (F1: {f1_v:.4f}) | Test: {acc_t:.4f} (F1: {f1_t:.4f})")
    
    return acc_v, acc_t, f1_v, f1_t, yt_pred, cv_scores.mean()

def main():
    ap = argparse.ArgumentParser(description="Entrenar clasificadores clásicos sobre features WPT - MEJORADO")
    ap.add_argument("--feat_dir", type=Path, default=Path("features"))
    ap.add_argument("--out_dir",  type=Path, default=Path("models"))
    ap.add_argument("--save_report", action="store_true", default=True)
    ap.add_argument("--cv_folds", type=int, default=5, help="Número de folds para validación cruzada")
    args = ap.parse_args()

    print("📊 Cargando características...")
    (Xtr, ytr), (Xv, yv), (Xt, yt) = load_features(args.feat_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📈 Dimensiones: Train {Xtr.shape}, Valid {Xv.shape}, Test {Xt.shape}")
    print(f"🎯 Número de clases: {len(np.unique(ytr))}")

    results = []

    # 1) Linear SVM con mejor configuración
    print("\n🔧 Entrenando LinearSVC...")
    svm = LinearSVC(
        dual=False, 
        C=0.1,  # Más regularización
        max_iter=2000,
        random_state=42
    )
    svm.fit(Xtr, ytr)
    acc_v, acc_t, f1_v, f1_t, yt_pred, cv_score = eval_model(
        "LinearSVC", svm, Xv, yv, Xt, yt, args.cv_folds
    )
    pickle.dump(svm, open(args.out_dir / "linear_svc.pkl", "wb"))
    results.append(("LinearSVC", acc_v, acc_t, f1_t, cv_score, yt_pred))

    # 2) Logistic Regression optimizado
    print("🔧 Entrenando LogisticRegression...")
    lr = LogisticRegression(
        max_iter=1000, 
        n_jobs=-1, 
        multi_class="multinomial",
        C=0.1,  # Más regularización
        random_state=42
    )
    lr.fit(Xtr, ytr)
    acc_v, acc_t, f1_v, f1_t, yt_pred, cv_score = eval_model(
        "LogisticRegression", lr, Xv, yv, Xt, yt, args.cv_folds
    )
    pickle.dump(lr, open(args.out_dir / "logreg.pkl", "wb"))
    results.append(("LogisticRegression", acc_v, acc_t, f1_t, cv_score, yt_pred))

    # 3) kNN optimizado
    print("🔧 Entrenando kNN...")
    knn = KNeighborsClassifier(
        n_neighbors=5, 
        n_jobs=-1,
        weights='distance'  # Mejor que 'uniform'
    )
    knn.fit(Xtr, ytr)
    acc_v, acc_t, f1_v, f1_t, yt_pred, cv_score = eval_model(
        "kNN(k=5)", knn, Xv, yv, Xt, yt, args.cv_folds
    )
    pickle.dump(knn, open(args.out_dir / "knn5.pkl", "wb"))
    results.append(("kNN(k=5)", acc_v, acc_t, f1_t, cv_score, yt_pred))

    # 4) Random Forest optimizado
    print("🔧 Entrenando RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=200,  # Menos árboles pero más profundos
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1, 
        random_state=42,
        class_weight='balanced'  # Importante si hay desbalance
    )
    rf.fit(Xtr, ytr)
    acc_v, acc_t, f1_v, f1_t, yt_pred, cv_score = eval_model(
        "RandomForest", rf, Xv, yv, Xt, yt, args.cv_folds
    )
    pickle.dump(rf, open(args.out_dir / "rf.pkl", "wb"))
    results.append(("RandomForest", acc_v, acc_t, f1_t, cv_score, yt_pred))

    # Resultados detallados
    print("\n" + "="*80)
    print("📊 RESULTADOS COMPARATIVOS")
    print("="*80)
    for name, acc_v, acc_t, f1_t, cv_score, _ in results:
        print(f"{name:20} | CV: {cv_score:.4f} | Valid: {acc_v:.4f} | Test: {acc_t:.4f} | F1-Test: {f1_t:.4f}")

    # Elegir el mejor por validación (priorizando F1-score)
    best_by_valid = max(results, key=lambda r: r[1])  # Por accuracy valid
    best_by_cv = max(results, key=lambda r: r[4])     # Por CV score
    best_by_f1 = max(results, key=lambda r: r[3])     # Por F1 test
    
    print(f"\n🏆 MEJOR POR VALIDACIÓN: {best_by_valid[0]} (Acc: {best_by_valid[1]:.4f})")
    print(f"🏆 MEJOR POR CV: {best_by_cv[0]} (CV Score: {best_by_cv[4]:.4f})")
    print(f"🏆 MEJOR POR F1: {best_by_f1[0]} (F1: {best_by_f1[3]:.4f})")

    # Reporte detallado del mejor por validación
    if args.save_report:
        best_name, _, _, _, _, best_yt_pred = best_by_valid
        
        # Reporte de clasificación
        rep = classification_report(yt, best_yt_pred, digits=4)
        cm = confusion_matrix(yt, best_yt_pred)
        
        # Guardar resultados
        np.save(args.out_dir / f"cm_{best_name}.npy", cm)
        (args.out_dir / f"report_{best_name}.txt").write_text(rep, encoding="utf-8")
        
        # Metadata completa
        meta = {
            "best_by_valid": best_by_valid[0],
            "best_by_cv": best_by_cv[0], 
            "best_by_f1": best_by_f1[0],
            "valid_acc": best_by_valid[1],
            "test_acc": best_by_valid[2],
            "test_f1": best_by_valid[3],
            "cv_score": best_by_valid[4],
            "all_results": {
                name: {
                    "valid_acc": acc_v, 
                    "test_acc": acc_t, 
                    "test_f1": f1_t,
                    "cv_score": cv_score
                } for name, acc_v, acc_t, f1_t, cv_score, _ in results
            }
        }
        
        with open(args.out_dir / "summary.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"\n💾 Reportes guardados en: {args.out_dir}")

if __name__ == "__main__":
    main()