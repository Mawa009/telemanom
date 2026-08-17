"""
Sweeps post-training error-thresholding parameters WITHOUT retraining --
reuses the already-saved y_hat predictions from a completed Detector.run(),
and only re-runs the fast Errors class (smoothing + pruning + thresholding)
for each parameter combination.

Tunable here (safe, no retrain needed):
    p               - pruning threshold (default 0.13)
    smoothing_perc   - EWMA smoothing window fraction (default 0.05)
    error_buffer     - neighboring points pulled into a sequence (default 100)
    window_size      - trailing batches for error calc (default 30)

NOT safe to tune here (these change model input shape -> need full retrain):
    l_s, n_predictions, layers, epochs, dropout, etc.

Usage (in Kaggle, after d.run() has already completed once):
    from sweep_params import sweep_params
    sweep_params(run_id=d.id, test_folder="test", labels_folder="labels",
                 p_values=[0.05, 0.1, 0.13, 0.2, 0.3])
"""
import numpy as np
import pandas as pd
import glob
import os
import copy

from telemanom.helpers import Config
from telemanom.channel import Channel
from telemanom.errors import Errors

from system_confusion_matrix1 import mask_to_sequences, sequence_level_metrics


def _load_gt_mask(test_folder, labels_folder):
    test_files = sorted(glob.glob(os.path.join(test_folder, "*.csv")))
    gt_mask = []
    for tf in test_files:
        base = os.path.basename(tf)
        label_file = os.path.join(labels_folder, base)
        label_df = pd.read_csv(label_file)
        gt_mask.append(label_df[label_df.columns[0]].values)
    return np.concatenate(gt_mask).astype(int)


def _run_with_params(run_id, config, chan_ids, n_rows, vote_pct=None):
    """Runs Errors (not Model) for every channel using already-saved y_hat,
    with whatever param values are currently set on `config`."""
    vote_count = np.zeros(n_rows, dtype=int)

    for chan_id in chan_ids:
        channel = Channel(config, chan_id)
        channel.load_data()
        channel.y_hat = np.load(os.path.join('data', run_id, 'y_hat',
                                              f'{chan_id}.npy'))

        errors = Errors(channel, config, run_id)
        errors.process_batches(channel)

        # errors.E_seq is ALREADY in raw test-row space (see PR #27 note
        # in system_confusion_matrix.py) -- do not add config.l_s again.
        for start, end in errors.E_seq:
            raw_start = max(0, start)
            raw_end = min(end, n_rows - 1)
            vote_count[raw_start:raw_end + 1] += 1

    n_channels = len(chan_ids)
    threshold = max(1, int(np.ceil((vote_pct or (1 / n_channels)) * n_channels)))
    pred = (vote_count >= threshold).astype(int)
    return pred


def sweep_params(run_id, test_folder, labels_folder, config_path='config.yaml',
                  p_values=None, smoothing_perc_values=None,
                  error_buffer_values=None, vote_pct=None):
    """
    Sweeps one parameter at a time (holding others at config.yaml defaults),
    reporting BOTH row-level and sequence-level metrics for each value.
    Pass a list for exactly ONE of p_values / smoothing_perc_values /
    error_buffer_values at a time to isolate its effect.
    """
    base_config = Config(config_path)
    gt_mask = _load_gt_mask(test_folder, labels_folder)
    n_rows = len(gt_mask)

    chan_ids = [os.path.splitext(f)[0] for f in os.listdir('data/test/')
                if f.endswith('.npy') and not f.startswith('_')]

    sweeps = []
    if p_values:
        sweeps.append(('p', p_values))
    if smoothing_perc_values:
        sweeps.append(('smoothing_perc', smoothing_perc_values))
    if error_buffer_values:
        sweeps.append(('error_buffer', error_buffer_values))

    if not sweeps:
        raise ValueError("Pass at least one of p_values / smoothing_perc_values / error_buffer_values")

    results = []

    for param_name, values in sweeps:
        print(f"\n=== Sweeping '{param_name}' (all else at config.yaml defaults) ===")
        for val in values:
            config = copy.deepcopy(base_config)
            setattr(config, param_name, val)

            pred = _run_with_params(run_id, config, chan_ids, n_rows, vote_pct)

            row_tp = int(np.sum((pred == 1) & (gt_mask == 1)))
            row_fp = int(np.sum((pred == 1) & (gt_mask == 0)))
            row_fn = int(np.sum((pred == 0) & (gt_mask == 1)))
            row_prec = row_tp / (row_tp + row_fp) if (row_tp + row_fp) > 0 else float('nan')
            row_rec = row_tp / (row_tp + row_fn) if (row_tp + row_fn) > 0 else float('nan')
            row_f1 = (2 * row_prec * row_rec / (row_prec + row_rec)
                      if (row_prec + row_rec) > 0 else float('nan'))

            seq = sequence_level_metrics(pred, gt_mask)

            print(f"  {param_name}={val}: "
                  f"ROW  TP={row_tp} FP={row_fp} FN={row_fn} P={row_prec:.3f} R={row_rec:.3f} F1={row_f1:.3f}  |  "
                  f"SEQ  TP={seq['tp']} FP={seq['fp']} FN={seq['fn']} P={seq['precision']:.3f} R={seq['recall']:.3f} F1={seq['f1']:.3f}")

            results.append({
                'param': param_name, 'value': val,
                'row_tp': row_tp, 'row_fp': row_fp, 'row_fn': row_fn,
                'row_precision': row_prec, 'row_recall': row_rec, 'row_f1': row_f1,
                'seq_tp': seq['tp'], 'seq_fp': seq['fp'], 'seq_fn': seq['fn'],
                'seq_precision': seq['precision'], 'seq_recall': seq['recall'], 'seq_f1': seq['f1'],
            })

    return pd.DataFrame(results)


def sweep_grid(run_id, test_folder, labels_folder, config_path='config.yaml',
                param_grid=None, vote_pct=None, sort_by='seq_f1'):
    """
    Joint grid search over MULTIPLE params at once (catches interaction
    effects that one-at-a-time sweeps can miss). Still no retraining --
    reuses saved y_hat, only reruns the fast Errors class per combo.

    param_grid: dict like {'p': [0.05, 0.13, 0.25], 'error_buffer': [50, 100, 150]}
        -- tries every combination (cartesian product). Keep grids small
        (e.g. 3x3 = 9 combos) since each combo still takes a few seconds
        per channel.

    sort_by: which column to sort results by, best first
        ('seq_f1', 'row_f1', 'seq_recall', 'row_recall', etc.)
    """
    import itertools

    base_config = Config(config_path)
    gt_mask = _load_gt_mask(test_folder, labels_folder)
    n_rows = len(gt_mask)

    chan_ids = [os.path.splitext(f)[0] for f in os.listdir('data/test/')
                if f.endswith('.npy') and not f.startswith('_')]

    param_names = list(param_grid.keys())
    value_lists = list(param_grid.values())
    combos = list(itertools.product(*value_lists))

    print(f"Testing {len(combos)} combinations across {param_names}...")

    results = []
    for combo in combos:
        config = copy.deepcopy(base_config)
        combo_dict = dict(zip(param_names, combo))
        for name, val in combo_dict.items():
            setattr(config, name, val)

        pred = _run_with_params(run_id, config, chan_ids, n_rows, vote_pct)

        row_tp = int(np.sum((pred == 1) & (gt_mask == 1)))
        row_fp = int(np.sum((pred == 1) & (gt_mask == 0)))
        row_fn = int(np.sum((pred == 0) & (gt_mask == 1)))
        row_prec = row_tp / (row_tp + row_fp) if (row_tp + row_fp) > 0 else float('nan')
        row_rec = row_tp / (row_tp + row_fn) if (row_tp + row_fn) > 0 else float('nan')
        row_f1 = (2 * row_prec * row_rec / (row_prec + row_rec)
                  if (row_prec + row_rec) > 0 else float('nan'))

        seq = sequence_level_metrics(pred, gt_mask)

        row = {**combo_dict,
               'row_tp': row_tp, 'row_fp': row_fp, 'row_fn': row_fn,
               'row_precision': row_prec, 'row_recall': row_rec, 'row_f1': row_f1,
               'seq_tp': seq['tp'], 'seq_fp': seq['fp'], 'seq_fn': seq['fn'],
               'seq_precision': seq['precision'], 'seq_recall': seq['recall'], 'seq_f1': seq['f1']}
        results.append(row)

    df = pd.DataFrame(results)
    sort_col = sort_by if sort_by in df.columns else 'seq_f1'
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    print(f"\nTop 5 combinations by {sort_col}:")
    display_cols = param_names + ['seq_tp', 'seq_fp', 'seq_fn', 'seq_f1', 'row_f1']
    print(df[display_cols].head(5).to_string(index=False))

    return df
