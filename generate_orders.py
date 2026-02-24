"""
generate_orders.py
Generates randomised stimulus orders for both the TRAIN and TEST runs,
matching the logic of the original eeg-things scripts.

Train output : sub-{id}_ses-{ses}_train.npy   shape (20, 56, 15)  dtype int
Test  output : sub-{id}_ses-{ses}_test.npy    shape (20, 51, 4)   dtype int

Encoding
--------
Training stimuli : integer = cat * 10 - 10 + img_num  
                   cat  : 1-indexed category  (1..1654)
                   img  : 1-indexed image within category (1..10)
Test stimuli     : integer = 1..200  (direct 1-indexed image number)
Target           : 0
"""

import numpy as np
import random
import os
import config as cfg


# ── Helpers ────────────────────────────────────────────────────────────────────

def shuffle(arr):
    """Return a shuffled copy without mutating the original."""
    arr = list(arr)
    random.shuffle(arr)
    return arr


def _target_sequences_and_positions(n_blocks, seqs_per_block, targets_per_block,
                                     img_per_sequence):
    """
    For each block, randomly choose which sequences contain a target and
    at which position (1-indexed, skipping position 0, matching MATLAB
    randi([2, img_per_sequence])).

    Returns
    -------
    target_sequences : list of lists, shape [n_blocks][targets_per_block]
    target_order     : flat list, length n_blocks * targets_per_block
    """
    target_sequences = []
    for _ in range(n_blocks):
        seqs = sorted(random.sample(range(seqs_per_block), targets_per_block))
        target_sequences.append(seqs)

    target_order = [
        random.randint(1, img_per_sequence - 1)   # 0-based idx; skips position 0
        for _ in range(n_blocks * targets_per_block)
    ]
    return target_sequences, target_order


def _fill_stim_matrix(train_order, target_sequences, target_order,
                       n_blocks, seqs_per_block, img_per_sequence,
                       targets_per_block):
    """
    Fill a 3-D integer matrix (img_per_sequence, seqs_per_block, n_blocks)
    from a flat train_order list, inserting 0s at target positions.
    """
    mat = np.zeros((img_per_sequence, seqs_per_block, n_blocks), dtype=int)

    train_counter        = 0
    target_counter_order = 0

    for b in range(n_blocks):
        target_counter_seq = 0
        for s in range(seqs_per_block):
            target_occurred = False
            for i in range(img_per_sequence):

                is_target_seq = (
                    target_counter_seq < targets_per_block
                    and s == target_sequences[b][target_counter_seq]
                )
                is_target_pos = (i == target_order[target_counter_order])

                if is_target_seq and is_target_pos and not target_occurred:
                    mat[i, s, b] = 0          # target
                    target_occurred = True
                    if target_counter_seq < targets_per_block - 1:
                        target_counter_seq += 1
                    if target_counter_order < len(target_order) - 1:
                        target_counter_order += 1
                else:
                    mat[i, s, b] = train_order[train_counter]
                    train_counter += 1

    return mat


# ── Train order ────────────────────────────────────────────────────────────────

def generate_train_orders(sub_id):
    """
    Generate train stimulus orders for all 4 sessions.
    Saves sub-{sub_id}_ses-{ses}_train.npy  shape (20, 56, 15).
    """
    TOT_CAT    = cfg.TOT_TRAINING_CAT
    HALF_CAT   = cfg.TRAINING_CAT          # 827
    IMG_PER    = cfg.IMG_PER_CAT           # 10
    TOT_N      = HALF_CAT * IMG_PER        # 8270
    REP        = cfg.REP_TRAINING          # 2
    IPS        = cfg.IMG_PER_SEQUENCE      # 20
    SPB        = cfg.TRAIN_SEQ_PER_BLOCK   # 56
    N_BLOCKS   = cfg.TRAIN_BLOCKS          # 15
    TARG_PB    = cfg.TARGETS_PER_BLOCK     # 6

    EXTRA = IPS * SPB * N_BLOCKS - TARG_PB * N_BLOCKS - TOT_N * REP

    # Two independent shuffles → 4 unique halves (matches MATLAB cat_1 / cat_2)
    cat_1 = shuffle(range(1, TOT_CAT + 1))
    cat_2 = shuffle(range(1, TOT_CAT + 1))

    sessions_cat = {
        1: cat_1[:HALF_CAT],
        2: cat_1[HALF_CAT:],
        3: cat_2[:HALF_CAT],
        4: cat_2[HALF_CAT:],
    }

    for ses in range(1, 5):
        ses_cats = sessions_cat[ses]

        # Build shuffled stimulus vector (matches MATLAB)
        vec_cat = [c for c in ses_cats for _ in range(IMG_PER)]
        vec_num = list(range(1, IMG_PER + 1)) * HALF_CAT
        vec_cat = [c for c in vec_cat for _ in range(REP)]
        vec_num = [n for n in vec_num for _ in range(REP)]

        idx     = shuffle(range(TOT_N * REP))
        vec_cat = [vec_cat[i] for i in idx]
        vec_num = [vec_num[i] for i in idx]

        # MATLAB encoding: cat*10 - 10 + img_num
        train_order = [c * 10 - 10 + n for c, n in zip(vec_cat, vec_num)]

        # Append extra trials
        extra = shuffle(train_order)[:EXTRA]
        train_order = train_order + extra

        # Target placement
        target_sequences, target_order = _target_sequences_and_positions(
            N_BLOCKS, SPB, TARG_PB, IPS
        )

        # Fill matrix → (20, 56, 15)
        stim_order = _fill_stim_matrix(
            train_order, target_sequences, target_order,
            N_BLOCKS, SPB, IPS, TARG_PB
        )

        path = os.path.join(cfg.ORDER_DIR, f'sub-{sub_id}_ses-{ses}_train.npy')
        np.save(path, stim_order)
        print(f"  [Train] ses-{ses} saved → {path}  shape={stim_order.shape}")


# ── Test order ─────────────────────────────────────────────────────────────────

def generate_test_orders(sub_id):
    """
    Generate test stimulus orders for all 4 sessions.
    Saves sub-{sub_id}_ses-{ses}_test.npy  shape (20, 51, 4).

    Test images are 1-indexed integers 1..200.
    Each of the 200 images appears rep_test (20) times per session,
    spread across 4 blocks of 51 sequences each.
    """
    N_TEST   = cfg.TEST_CAT               # 200
    REP      = cfg.REP_TEST               # 20
    IPS      = cfg.IMG_PER_SEQUENCE       # 20
    SPB      = cfg.TEST_SEQ_PER_BLOCK     # 51
    N_BLOCKS = cfg.TEST_RUNS              # 4
    TARG_PB  = cfg.TARGETS_PER_BLOCK      # 6

    EXTRA = IPS * SPB * N_BLOCKS - TARG_PB * N_BLOCKS - N_TEST * REP

    # Build the full pool (1-indexed image numbers, each repeated REP times)
    test_pool = shuffle(list(range(1, N_TEST + 1)) * REP)

    # Append extra trials to fill exactly
    extra = shuffle(test_pool)[:EXTRA]
    test_order = test_pool + extra

    # Same pool is reused across all 4 sessions (test images are session-invariant)
    # but shuffled independently per session
    for ses in range(1, 5):
        ses_order = shuffle(test_order)

        target_sequences, target_order = _target_sequences_and_positions(
            N_BLOCKS, SPB, TARG_PB, IPS
        )

        stim_order = _fill_stim_matrix(
            ses_order, target_sequences, target_order,
            N_BLOCKS, SPB, IPS, TARG_PB
        )

        path = os.path.join(cfg.ORDER_DIR, f'sub-{sub_id}_ses-{ses}_test.npy')
        np.save(path, stim_order)
        print(f"  [Test]  ses-{ses} saved → {path}  shape={stim_order.shape}")


# ── Entry point ────────────────────────────────────────────────────────────────

def create_orders():
    os.makedirs(cfg.ORDER_DIR, exist_ok=True)
    sub_id = input("Enter Subject ID: ").strip()
    print(f"\nGenerating orders for subject {sub_id}...")
    generate_train_orders(sub_id)
    generate_test_orders(sub_id)
    print("\nDone.")


if __name__ == "__main__":
    create_orders()
