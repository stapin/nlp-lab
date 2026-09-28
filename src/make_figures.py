"""Dataset statistics table + figures for the report (report/figures/*.pdf)."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGS, RESULTS, load_split  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order (light)
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f3f7fd", "#2a78d6", "#0b3a75"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})

ORDER = ["human", "gpt-4-turbo", "llama-3.3-70b", "gemma-2-27b", "unknown"]


def dataset_stats():
    rows = []
    frames = {s: load_split(s) for s in ["train", "val", "dev"]}
    for name in ["aux", "heldout"]:
        try:
            frames[name] = load_split(name)
        except FileNotFoundError:
            pass
    for split, df in frames.items():
        key = "generator" if "generator" in df else "label"
        for lab, d in df.groupby(key):
            n = d.text.str.split().str.len()
            rows.append({"split": split, "class": lab, "texts": len(d), "words_mean": n.mean(),
                         "words_median": n.median(), "digits_per_100w": 100 * d.text.str.count(r"\d").sum() / n.sum()})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "dataset_stats.csv", index=False)
    print(out.round(1).to_string(index=False))


def length_figure():
    dev = load_split("dev")
    dev["label"] = dev.label.replace({"unknown": "unknown\n(GigaChat-Lite)"})
    order = [o if o != "unknown" else "unknown\n(GigaChat-Lite)" for o in ORDER]
    data = [dev.loc[dev.label == o, "text"].str.split().str.len().clip(upper=400).values for o in order]
    try:
        held = load_split("heldout")
        for g in held.generator.unique():
            if g != "human":
                order.append(f"{g}\n(ours, held-out)")
                data.append(held.loc[held.generator == g, "text"].str.split().str.len().values)
    except FileNotFoundError:
        pass
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(order) + 0.6))
    bp = ax.boxplot(data[::-1], orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=INK2),
                    capprops=dict(color=INK2), boxprops=dict(edgecolor=INK2, linewidth=0.8))
    for patch, lab in zip(bp["boxes"], order[::-1]):
        patch.set_facecolor(SERIES[1] if lab == "human" else SERIES[0])
        patch.set_alpha(0.85)
    ax.set_yticks(range(1, len(order) + 1), order[::-1])
    ax.set_xlabel("Abstract length, words (dev set; whiskers = 1.5 IQR)")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIGS / "lengths.pdf")
    fig.savefig(FIGS / "lengths.png")


def length_bucket_figure():
    """Binary AUROC per length bucket on dev for the main methods."""
    from sklearn.metrics import roc_auc_score
    methods = {
        "TF-IDF + LR": "tfidf__full",
        "Binoculars (zero-shot)": "zeroshot-binoculars__full",
        "mDeBERTa-v3": "mdeberta__full",
        "Hybrid (ours)": "hybrid-lr[tfidf+lm+style]__full",
    }
    dev = load_split("dev")
    n = dev.text.str.split().str.len().values
    bins = [0, 50, 75, 100, 10_000]
    names = ["<50", "50-74", "75-99", "100+"]
    b = np.digitize(n, bins[1:-1])
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    width = 0.2
    for k, (label, f) in enumerate(methods.items()):
        path = RESULTS / "preds" / f"{f}.parquet"
        if not path.exists():
            continue
        p = pd.read_parquet(path).query("split == 'dev'")
        ai, score = (p.label != "human").values, 1 - p.p_human.values
        auc = [roc_auc_score(ai[b == i], score[b == i]) for i in range(len(names))]
        ax.bar(np.arange(len(names)) + (k - 1.5) * width, auc, width - 0.03, color=SERIES[k], label=label)
    ai_all = (dev.label != "human").values
    ticks = [f"{nm}\nAI {(ai_all[b == i]).sum()}\nhuman {(~ai_all[b == i]).sum()}" for i, nm in enumerate(names)]
    ax.set_xticks(range(len(names)), ticks, fontsize=7)
    ax.set_xlabel("Abstract length bucket, words (dev set)")
    ax.set_ylabel("Human-vs-AI ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(FIGS / "length_buckets.pdf")
    fig.savefig(FIGS / "length_buckets.png")


def confusion_figure(pred_file="hybrid-lr[tfidf+lm+style]+aux__full"):
    from evaluate import decide
    path = RESULTS / "preds" / f"{pred_file}.parquet"
    if not path.exists():
        return
    d = pd.read_parquet(path).query("split == 'dev'")
    pred = decide(d)
    cm = pd.crosstab(pd.Categorical(d.label, ORDER), pd.Categorical(pred, ORDER), normalize="index", dropna=False)
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.imshow(cm.values, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(ORDER)), ORDER, rotation=35, ha="right")
    ax.set_yticks(range(len(ORDER)), ORDER)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    fig.savefig(FIGS / "confusion.pdf")
    fig.savefig(FIGS / "confusion.png")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    dataset_stats()
    length_figure()
    length_bucket_figure()
    confusion_figure()
