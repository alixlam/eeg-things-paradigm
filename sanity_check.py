"""
sanity_check.py
Verifies stimulus order files at two levels:
  1. Per-session: correct structure and repetitions within each session
  2. Across all sessions: every image seen correct number of times total
        Train: 1654 categories × 10 images × 4 reps = 16540 unique (cat,img) pairs,
               each appearing exactly 4 times across the 4 sessions
               (2 reps per session × 2 sessions each image appears in)
        Test:  200 images × 80 reps total across 4 sessions (20 per session)

Usage:
    python sanity_check.py --sub 1
"""

import numpy as np
import argparse
import os
from collections import Counter
import config as cfg

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
args = parser.parse_args()

sub_id = args.sub

def check(cond, msg_ok, msg_fail):
    if cond:
        print(f"    ✓ {msg_ok}")
    else:
        print(f"    ✗ {msg_fail}")

print(f"\n{'='*60}")
print(f"Sanity check — subject {sub_id}")
print(f"{'='*60}")


# ── TRAIN ──────────────────────────────────────────────────────────────────────
print("\n── TRAIN: per session ─────────────────────────────────────")

all_train_counts = Counter()   # accumulate across all sessions

for ses in range(1, 5):
    path = os.path.join(cfg.ORDER_DIR, f'sub-{sub_id}_ses-{ses}_train.npy')
    if not os.path.exists(path):
        print(f"\n  ses-{ses}: MISSING {path}")
        continue

    order     = np.load(path)            # (20, 56, 15)
    flat      = order.flatten()
    stim_vals = flat[flat != 0]

    counts    = Counter(stim_vals.tolist())
    all_train_counts += counts

    n_unique   = len(counts)
    min_count  = min(counts.values())
    max_count  = max(counts.values())
    cats_used  = set((v - 1) // cfg.IMG_PER_CAT + 1 for v in counts)
    invalid    = [v for v in counts if v < 1 or
                  v > cfg.TOT_TRAINING_CAT * cfg.IMG_PER_CAT]

    print(f"\n  ses-{ses}:")
    print(f"    Total presentations (excl. targets) : {len(stim_vals)}")
    check(n_unique == cfg.TRAINING_CAT * cfg.IMG_PER_CAT,
          f"Unique images: {n_unique}",
          f"Unique images: {n_unique}, expected {cfg.TRAINING_CAT * cfg.IMG_PER_CAT}")
    check(len(cats_used) == cfg.TRAINING_CAT,
          f"Categories used: {len(cats_used)}",
          f"Categories used: {len(cats_used)}, expected {cfg.TRAINING_CAT}")
    check(min_count >= cfg.REP_TRAINING,
          f"Min reps per image: {min_count} (>= {cfg.REP_TRAINING})",
          f"Min reps per image: {min_count}, expected >= {cfg.REP_TRAINING}")
    print(f"    Max reps per image                  : {max_count}")
    check(len(invalid) == 0,
          "No invalid stim_vals",
          f"Invalid stim_vals found: {invalid[:5]}")

# ── TRAIN: across all sessions ────────────────────────────────────────────────
print("\n── TRAIN: across all 4 sessions ───────────────────────────")

# Each of the 1654*10 = 16540 images should appear exactly
# REP_TRAINING (2) times per session × 2 sessions = 4 times total
expected_total_unique = cfg.TOT_TRAINING_CAT * cfg.IMG_PER_CAT   # 16540
expected_total_reps   = cfg.REP_TRAINING * 2                      # 4

n_unique_total  = len(all_train_counts)
min_total       = min(all_train_counts.values())
max_total       = max(all_train_counts.values())
wrong_reps      = {k: v for k, v in all_train_counts.items()
                   if v != expected_total_reps}

print(f"    Total unique images seen            : {n_unique_total}")
check(n_unique_total == expected_total_unique,
      f"All {expected_total_unique} images present",
      f"Got {n_unique_total}, expected {expected_total_unique}")
check(min_total >= expected_total_reps,
      f"Min total reps: {min_total} (>= {expected_total_reps})",
      f"Min total reps: {min_total}, expected >= {expected_total_reps}")
check(max_total <= expected_total_reps + 2,
      f"Max total reps: {max_total} (reasonable)",
      f"Max total reps: {max_total} — some images shown too many times")
check(len(wrong_reps) == 0,
      f"All images seen exactly {expected_total_reps} times",
      f"{len(wrong_reps)} images not seen exactly {expected_total_reps} times "
      f"(first few: { {k: wrong_reps[k] for k in list(wrong_reps)[:5]} })")


# ── TEST ───────────────────────────────────────────────────────────────────────
print("\n── TEST: per session ──────────────────────────────────────")

all_test_counts = Counter()

for ses in range(1, 5):
    path = os.path.join(cfg.ORDER_DIR, f'sub-{sub_id}_ses-{ses}_test.npy')
    if not os.path.exists(path):
        print(f"\n  ses-{ses}: MISSING {path}")
        continue

    order     = np.load(path)            # (20, 51, 4)
    flat      = order.flatten()
    stim_vals = flat[flat != 0]

    counts   = Counter(stim_vals.tolist())
    all_test_counts += counts

    n_unique  = len(counts)
    min_count = min(counts.values())
    max_count = max(counts.values())
    invalid   = [v for v in counts if v < 1 or v > cfg.TEST_CAT]

    print(f"\n  ses-{ses}:")
    print(f"    Total presentations (excl. targets) : {len(stim_vals)}")
    check(n_unique == cfg.TEST_CAT,
          f"Unique images: {n_unique}",
          f"Unique images: {n_unique}, expected {cfg.TEST_CAT}")
    check(min_count >= cfg.REP_TEST,
          f"Min reps per image: {min_count} (>= {cfg.REP_TEST})",
          f"Min reps per image: {min_count}, expected >= {cfg.REP_TEST}")
    print(f"    Max reps per image                  : {max_count}")
    check(len(invalid) == 0,
          "No invalid stim_vals",
          f"Invalid stim_vals found: {invalid[:5]}")

# ── TEST: across all sessions ─────────────────────────────────────────────────
print("\n── TEST: across all 4 sessions ────────────────────────────")

# Each of the 200 images should appear REP_TEST (20) × 4 sessions = 80 times
expected_total_reps = cfg.REP_TEST * 4   # 80
wrong_reps_test     = {k: v for k, v in all_test_counts.items()
                       if v != expected_total_reps}

n_unique_total = len(all_test_counts)
min_total      = min(all_test_counts.values())
max_total      = max(all_test_counts.values())

print(f"    Total unique test images seen       : {n_unique_total}")
check(n_unique_total == cfg.TEST_CAT,
      f"All {cfg.TEST_CAT} images present",
      f"Got {n_unique_total}, expected {cfg.TEST_CAT}")
check(min_total >= expected_total_reps,
      f"Min total reps: {min_total} (>= {expected_total_reps})",
      f"Min total reps: {min_total}, expected >= {expected_total_reps}")
check(max_total <= expected_total_reps + 4,
      f"Max total reps: {max_total} (reasonable)",
      f"Max total reps: {max_total} — some images shown too many times")
check(len(wrong_reps_test) == 0,
      f"All images seen exactly {expected_total_reps} times",
      f"{len(wrong_reps_test)} images not seen exactly {expected_total_reps} times "
      f"(first few: { {k: wrong_reps_test[k] for k in list(wrong_reps_test)[:5]} })")

print(f"\n{'='*60}\n")
