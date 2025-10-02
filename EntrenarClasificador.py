import argparse, pickle, json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def load_features(root: Path):
    X_train = np.load(root / "X_train.npy"); y_train = np.load(root / "y_train.npy")
    X_valid = np.load(root / "X_valid.npy"); y_valid = np.load(root / "y_valid.npy")
    X_test  = np.load(root / "X_test.npy");  y_test  = np.load(root / "y_test.npy")
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def eval_model(name, model, Xv, yv, Xt, yt):
    yv_pred = model.predict(Xv); yt_pred = model.predict(Xt)
    acc_v = accuracy_score(yv, yv_pred); acc_t = accuracy_score(yt, yt_pred)
    print(f"\n{name}  |  valid acc: {acc_v:.4f}  |  test acc: {acc_t:.4f}")
    return acc_v, acc_t, yt_pred

def main():
    ap = argparse.ArgumentParser(description="Entrenar clasificadores clásicos sobre features WPT")
    ap.add_argument("--feat_dir", type=Path, default=Path("features_wpt"))
    ap.add_argument("--out_dir",  type=Path, default=Path("models"))
    ap.add_argument("--save_report", action="store_true")
    args = ap.parse_args()

    (Xtr, ytr), (Xv, yv), (Xt, yt) = load_features(args.feat_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 1) Linear SVM (rápido y fuerte con muchas features)
    svm = LinearSVC(dual=False, C=1.0)
    svm.fit(Xtr, ytr)
    acc_v, acc_t, yt_pred = eval_model("LinearSVC", svm, Xv, yv, Xt, yt)
    pickle.dump(svm, open(args.out_dir / "linear_svc.pkl", "wb"))
    results.append(("LinearSVC", acc_v, acc_t, yt_pred))

    # 2) Logistic Regression (multinomial)
    lr = LogisticRegression(max_iter=200, n_jobs=-1, multi_class="multinomial", C=1.0)
    lr.fit(Xtr, ytr)
    acc_v, acc_t, yt_pred = eval_model("LogisticRegression", lr, Xv, yv, Xt, yt)
    pickle.dump(lr, open(args.out_dir / "logreg.pkl", "wb"))
    results.append(("LogisticRegression", acc_v, acc_t, yt_pred))

    # 3) kNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(Xtr, ytr)
    acc_v, acc_t, yt_pred = eval_model("kNN(k=5)", knn, Xv, yv, Xt, yt)
    pickle.dump(knn, open(args.out_dir / "knn5.pkl", "wb"))
    results.append(("kNN(k=5)", acc_v, acc_t, yt_pred))

    # 4) Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=0)
    rf.fit(Xtr, ytr)
    acc_v, acc_t, yt_pred = eval_model("RandomForest", rf, Xv, yv, Xt, yt)
    pickle.dump(rf, open(args.out_dir / "rf.pkl", "wb"))
    results.append(("RandomForest", acc_v, acc_t, yt_pred))

    # Elegir el mejor por valid
    best = max(results, key=lambda r: r[1])
    best_name, best_acc_v, best_acc_t, best_yt_pred = best
    print(f"\n🏆 Mejor por valid: {best_name}  |  valid {best_acc_v:.4f}  |  test {best_acc_t:.4f}")

    # Reporte opcional
    if args.save_report:
        cm = confusion_matrix(yt, best_yt_pred)
        rep = classification_report(yt, best_yt_pred, digits=4)
        np.save(args.out_dir / f"cm_{best_name}.npy", cm)
        (args.out_dir / f"report_{best_name}.txt").write_text(rep, encoding="utf-8")
        meta = {"best_model": best_name, "valid_acc": best_acc_v, "test_acc": best_acc_t}
        (args.out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
