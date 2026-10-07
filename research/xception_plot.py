"""
Plots for the DeepfakeBench Xception detector, including the Effort paper's analyses.

Put this file next to your xception.py (the standalone script), then run:
  python3 analyze_xception.py --weights xception_best.pth --real real --fake fake

It runs the model ONCE, caches everything in plots/outputs.npz, and writes:
  1_score_histogram.png   P(fake) distributions for real vs fake
  2_roc_curve.png         ROC curve with AUC, and the best threshold marked
  3_confusion_matrix.png  errors at threshold 0.5 and at the best threshold
  4_tsne.png              t-SNE of the 2048-d embeddings    (paper Fig. 2)
  5_pca_variance.png      PCA explained variance            (paper Fig. 3)
  6_logit_scatter.png     real-logit vs fake-logit          (paper Fig. 7)
  misclassified.csv       every wrong image with its score, worst first
Use --recompute to ignore the cache after changing the data.
"""
import argparse, csv, os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")                     # save files, no window needed
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from xception import build_model, preprocess, list_images   # your standalone script

REAL_C, FAKE_C = "#2b83ba", "#d7191c"


@torch.no_grad()
def extract(model, paths, device, bs=32):
    """Return logits [N,2] and 2048-d embeddings [N,2048] for every image."""
    logits, embs = [], []
    for i in range(0, len(paths), bs):
        x = torch.stack([preprocess(p) for p in paths[i:i + bs]]).to(device)
        out, _ = model(x)
        logits.append(out.cpu())
        embs.append(model.last_emb.cpu())   # set inside model.classifier()
        print(f"  {min(i + bs, len(paths))}/{len(paths)}", end="\r")
    print()
    return torch.cat(logits).numpy(), torch.cat(embs).numpy()


def load_or_compute(a, cache):
    if os.path.exists(cache) and not a.recompute:
        print(f"[cache] using {cache}")
        d = np.load(cache, allow_pickle=True)
        return d["paths"], d["labels"], d["logits"], d["embs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(a.weights, device)
    real, fake = list_images(a.real), list_images(a.fake)
    paths = np.array(real + fake)
    labels = np.array([0] * len(real) + [1] * len(fake))
    print(f"running model on {len(paths)} images ({device})")
    logits, embs = extract(model, list(paths), device)
    np.savez(cache, paths=paths, labels=labels, logits=logits, embs=embs)
    return paths, labels, logits, embs


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)
    print(f"  wrote {name}")


def plot_hist(scores, labels, out, best_t):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 26)
    ax.hist(scores[labels == 0], bins, alpha=0.6, color=REAL_C, label="real")
    ax.hist(scores[labels == 1], bins, alpha=0.6, color=FAKE_C, label="fake")
    ax.axvline(0.5, color="k", ls="--", lw=1, label="threshold 0.5")
    ax.axvline(best_t, color="green", ls=":", lw=1.5, label=f"best threshold {best_t:.2f}")
    ax.set(xlabel="P(fake)", ylabel="number of images", title="Score distribution")
    ax.legend()
    save(fig, out, "1_score_histogram.png")


def plot_roc(scores, labels, out):
    fpr, tpr, thr = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    j = np.argmax(tpr - fpr)                     # Youden's J: best balance of TPR and FPR
    best_t = float(min(thr[j], 1.0))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=FAKE_C, lw=2, label=f"Xception (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guessing")
    ax.scatter(fpr[j], tpr[j], color="green", zorder=3, label=f"best threshold {best_t:.2f}")
    ax.set(xlabel="False positive rate (reals called fake)",
           ylabel="True positive rate (fakes caught)", title="ROC curve")
    ax.legend(loc="lower right")
    save(fig, out, "2_roc_curve.png")
    return best_t, auc


def plot_confusion(scores, labels, out, best_t):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, t in zip(axes, (0.5, best_t)):
        cm = confusion_matrix(labels, scores >= t, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=14,
                    color="white" if v > cm.max() / 2 else "black")
        acc = np.trace(cm) / cm.sum()
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["real", "fake"],
               yticklabels=["real", "fake"], xlabel="predicted", ylabel="actual",
               title=f"threshold {t:.2f}  (accuracy {acc:.1%})")
    save(fig, out, "3_confusion_matrix.png")


def plot_tsne(embs, labels, out):
    perplexity = min(30, max(5, (len(embs) - 1) // 3))   # must be < number of samples
    z = TSNE(n_components=2, perplexity=perplexity, init="pca",
             random_state=0).fit_transform(embs)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for lab, name, c in ((0, "real", REAL_C), (1, "fake", FAKE_C)):
        ax.scatter(*z[labels == lab].T, s=14, alpha=0.7, color=c, label=name)
    ax.set(title="t-SNE of 2048-d embeddings (cf. paper Fig. 2)", xticks=[], yticks=[])
    ax.legend()
    save(fig, out, "4_tsne.png")


def plot_pca(embs, out):
    n = min(50, embs.shape[0], embs.shape[1])
    ratio = PCA(n_components=n).fit(embs).explained_variance_ratio_
    cum = np.cumsum(ratio)
    k90 = int(np.searchsorted(cum, 0.90) + 1) if cum[-1] >= 0.90 else None
    k_txt = f"{k90} components explain 90%" if k90 else f"top {n} explain only {cum[-1]:.0%}"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(1, n + 1), ratio, color="gray", label="per component")
    ax.plot(range(1, n + 1), cum, "o-", ms=3, color=FAKE_C, label="cumulative")
    ax.axhline(0.9, color="k", ls="--", lw=1)
    ax.set(xlabel="principal component", ylabel="explained variance ratio",
           title=f"PCA of embeddings: {k_txt} (cf. paper Fig. 3)",
           ylim=(0, 1.02))
    ax.legend()
    save(fig, out, "5_pca_variance.png")
    return k90, ratio[:2].sum()


def plot_logits(logits, labels, out):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for lab, name, c in ((0, "real", REAL_C), (1, "fake", FAKE_C)):
        ax.scatter(logits[labels == lab, 0], logits[labels == lab, 1],
                   s=14, alpha=0.7, color=c, label=name)
    # Fit y = a*x + b over all points; the paper reports a slope close to -1.
    a, b = np.polyfit(logits[:, 0], logits[:, 1], 1)
    r = np.corrcoef(logits[:, 0], logits[:, 1])[0, 1]
    xs = np.linspace(logits[:, 0].min(), logits[:, 0].max(), 50)
    ax.plot(xs, a * xs + b, "k--", lw=1, label=f"fit: y = {a:.2f}x + {b:.2f} (r = {r:.3f})")
    ax.set(xlabel="logit for class 0 (real)", ylabel="logit for class 1 (fake)",
           title="Real vs fake logits (cf. paper Fig. 7)")
    ax.legend()
    save(fig, out, "6_logit_scatter.png")
    return a, r


def write_errors(paths, labels, scores, out, t=0.5):
    wrong = np.where((scores >= t) != labels.astype(bool))[0]
    # Worst first: how far the score is on the wrong side of the threshold.
    wrong = wrong[np.argsort(-np.abs(scores[wrong] - t))]
    with open(os.path.join(out, "misclassified.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "true_label", "p_fake"])
        for i in wrong:
            w.writerow([paths[i], "fake" if labels[i] else "real", f"{scores[i]:.4f}"])
    print(f"  wrote misclassified.csv ({len(wrong)} images)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--fake", required=True)
    ap.add_argument("--out", default="plots")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    paths, labels, logits, embs = load_or_compute(a, os.path.join(a.out, "outputs.npz"))
    scores = torch.softmax(torch.from_numpy(logits), 1)[:, 1].numpy()

    print("making plots:")
    best_t, auc = plot_roc(scores, labels, a.out)
    plot_hist(scores, labels, a.out, best_t)
    plot_confusion(scores, labels, a.out, best_t)
    plot_tsne(embs, labels, a.out)
    k90, top2 = plot_pca(embs, a.out)
    slope, r = plot_logits(logits, labels, a.out)
    write_errors(paths, labels, scores, a.out)

    print("\nsummary")
    print(f"  AUC                          {auc:.4f}")
    print(f"  best threshold               {best_t:.3f}")
    print(f"  PCA comps for 90% variance   {k90 or '>50'}   (top-2 explain {top2:.1%})")
    print(f"  logit fit slope / corr       {slope:.3f} / {r:.3f}   (paper: slope ~ -1)")


if __name__ == "__main__":
    main()