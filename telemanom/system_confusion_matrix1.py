
"""
Builds ONE system-level confusion matrix across all channels combined,
instead of telemanom's default per-channel scoring.

Logic:
- For each channel, take its detected anomaly windows (E_seq), which are
  ALREADY in raw test-row space (telemanom's errors.py applies the l_s
  offset internally when building E_seq -- see the comment in
  Errors.process_batches referencing PR #27). Do NOT re-apply l_s here.
- OR all channels together: a raw row is "system anomalous" if ANY
  (or, with vote_pct, a configurable fraction of) channels flagged it
- Compare that single combined array against the raw ground-truth mask
  (the original 0/1 column(s) from your labels/ folder), row by row

Run this AFTER Detector.run() has already produced results (so trained
models / y_hat / smoothed_errors already exist on disk for run_id).
"""
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

from telemanom.helpers import Config
from telemanom.channel import Channel
from telemanom.errors import Errors


def plot_confusion_matrix(tp, fp, fn, tn, save_path='system_confusion_matrix.png'):
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Normal', 'Predicted Anomaly'],
                yticklabels=['Actual Normal', 'Actual Anomaly'], ax=ax)
    ax.set_title('System-Level Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"Saved plot to {save_path}")


def build_system_confusion_matrix(run_id, test_folder, labels_folder,
                                   config_path='config.yaml',
                                   vote_pct=None, sweep=True):
    """
    vote_pct: if set (e.g. 0.2 for 20%), a row is "system anomalous" only
        if at least that fraction of channels flagged it. If None, uses
        the pure OR logic (any 1 channel triggers -- equivalent to a
        vote_pct just above 0).
    sweep: if True, also prints/plots precision-recall across a range of
        vote_pct values so you can pick one grounded in your actual data.
    """
    config = Config(config_path)

    test_files = sorted(glob.glob(os.path.join(test_folder, "*.csv")))
    gt_mask = []
    for tf in test_files:
        base = os.path.basename(tf)
        label_file = os.path.join(labels_folder, base)
        label_df = pd.read_csv(label_file)
        gt_mask.append(label_df[label_df.columns[0]].values)
    gt_mask = np.concatenate(gt_mask).astype(int)
    n_rows = len(gt_mask)

    chan_ids = [os.path.splitext(f)[0] for f in os.listdir('data/test/')
                if f.endswith('.npy') and not f.startswith('_')]
    n_channels = len(chan_ids)

    vote_count = np.zeros(n_rows, dtype=int)

    for chan_id in chan_ids:
        channel = Channel(config, chan_id)
        channel.load_data()
        channel.y_hat = np.load(os.path.join('data', run_id, 'y_hat',
                                              f'{chan_id}.npy'))

        errors = Errors(channel, config, run_id)
        errors.process_batches(channel)

        # errors.E_seq is ALREADY in raw test-row space -- Errors.process_batches
        # adds the l_s offset internally (see the "PR #27" comment in errors.py).
        # Do NOT add config.l_s again here, or every detection ends up shifted
        # forward by l_s rows relative to the ground-truth mask.
        for start, end in errors.E_seq:
            raw_start = max(0, start)
            raw_end = min(end, n_rows - 1)
            vote_count[raw_start:raw_end + 1] += 1

        print(f"{chan_id}: {len(errors.E_seq)} anomaly window(s) detected")

    def metrics_at(pct):
        threshold = max(1, int(np.ceil(pct * n_channels))) if pct else 1
        pred = (vote_count >= threshold).astype(int)
        tp = int(np.sum((pred == 1) & (gt_mask == 1)))
        fp = int(np.sum((pred == 1) & (gt_mask == 0)))
        fn = int(np.sum((pred == 0) & (gt_mask == 1)))
        tn = int(np.sum((pred == 0) & (gt_mask == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        rec = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else float('nan')
        return threshold, tp, fp, fn, tn, prec, rec, f1

    if sweep:
        print(f"\n=== Precision/Recall across vote thresholds (out of {n_channels} channels) ===")
        for pct in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            thr, tp, fp, fn, tn, prec, rec, f1 = metrics_at(pct)
            print(f"{pct:>5.0%} (>= {thr}/{n_channels} channels): "
                  f"TP={tp} FP={fp} FN={fn} | Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

    chosen_pct = vote_pct if vote_pct is not None else (1 / n_channels)
    threshold, tp, fp, fn, tn, precision, recall, f1 = metrics_at(chosen_pct)

    print(f"\n=== FINAL (using threshold: {chosen_pct:.0%} => >= {threshold}/{n_channels} channels) ===")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    plot_confusion_matrix(tp, fp, fn, tn)

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "vote_count": vote_count, "gt_mask": gt_mask,
            "threshold_channels": threshold, "n_channels": n_channels}


if __name__ == "__main__":
    import sys
    run_id = sys.argv[1]
    build_system_confusion_matrix(run_id, "test", "labels")