"""
main.py
PsychoPy data collection — THINGS-EEG paradigm (train + test).

Run numbering
-------------
Runs 1–4  → TEST   (51 sequences × 1 block,  order from _test.npy)
Runs 5–15  → TRAIN (56 sequences × 1 block,  order from _train.npy)

Stimulus encoding (matches generate_orders.py / MATLAB)
--------------------------------------------------------
Train  : integer = cat * 10 - 10 + img_num
Test   : integer = 1..200
Target : 0

Triggers
--------
TARGET = 255  (catch trial — Buzz Lightyear)
Test   : trigger = image index (1..200)
Train  : trigger = category index (1..827, mod 254 + 1 to fit in a byte)

Photodiode square
-----------------
White square drawn top-right corner on every image ON flip, black on OFF.
Combined with parallel-port trigger for millisecond-accurate timing checks.

Timing logic
------------
Each trial:
  1. Pre-load next image into buffer BEFORE the flip deadline
  2. Flip ON at t_prev_onset + SOA (scheduled deadline)
  3. Send trigger
  4. Wait IMG_DUR
  5. Flip OFF (blank + fixation)
  6. Remaining gap before next SOA is used to load the next image

This ensures:
  - ON flips are spaced exactly SOA apart
  - Image is on screen for exactly IMG_DUR
  - Image loading happens during the blank period, not during the timed window

DEBUG mode
----------
- Windowed (1024×768), any monitor
- No parallel port
- Everything else (images, timing, logging) is identical to the real task
- If target folder is missing, blue placeholder is shown instead
"""

from psychopy import visual, core, event, gui, parallel
import numpy as np
import pandas as pd
import glob
import re
import os
import config as cfg
import platform


# ── Image lookup ───────────────────────────────────────────────────────────────

def build_image_lookup():
    """
    Pre-build {stim_val: filepath} dicts for test and train images.
    Call once at startup — avoids glob calls inside the trial loop.

    Test folder structure:
        test_images/00001_aircraft_carrier/aircraft_carrier_06s.jpg
        stim_val = folder index (1..200)

    Train folder structure:
        training_images/00001_aardvark/aardvark_01b.jpg  → stim_val = 1
                                       aardvark_02s.jpg  → stim_val = 2
        stim_val = cat * 10 - 10 + img_num
    """
    # ── Test: one image per folder ────────────────────────────────────────────
    test_lookup = {}
    for folder in sorted(glob.glob(os.path.join(cfg.IMG_DIR, 'test_images', '*'))):
        idx = int(os.path.basename(folder).split('_')[0])
        images = [p for p in glob.glob(os.path.join(folder, '*.jpg'))
                  if not os.path.basename(p).startswith('._')]
        if not images:
            raise FileNotFoundError(f"No image in test folder: {folder}")
        test_lookup[idx] = images[0]

    # ── Train: 10 images per folder, sorted by number embedded in filename ────
    train_lookup = {}
    for folder in sorted(glob.glob(os.path.join(cfg.IMG_DIR, 'training_images', '*'))):
        cat = int(os.path.basename(folder).split('_')[0])
        images = sorted(
            [p for p in glob.glob(os.path.join(folder, '*.jpg'))
             if not os.path.basename(p).startswith('._')],
            key=lambda p: int(re.search(r'_(\d+)', os.path.basename(p)).group(1))
        )
        # aardvark_01b.jpg → 1, aardvark_02s.jpg → 2, etc.
        if len(images) != cfg.IMG_PER_CAT:
            raise ValueError(
                f"Expected {cfg.IMG_PER_CAT} images in {folder}, found {len(images)}"
            )
        for img_num, path in enumerate(images, start=1):
            stim_val = cat * 10 - 10 + img_num
            train_lookup[stim_val] = path

    print(f"Image lookup built: {len(test_lookup)} test, {len(train_lookup)} train.")
    return test_lookup, train_lookup


# ── Stimulus helpers ───────────────────────────────────────────────────────────

def decode_train_stim(val):
    """Decode MATLAB-style integer → (cat, img_num), both 1-indexed."""
    cat     = (val - 1) // cfg.IMG_PER_CAT + 1
    img_num = val - (cat - 1) * cfg.IMG_PER_CAT
    return cat, img_num


def get_trigger_code(val, is_test_run):
    """Return EEG trigger byte for a stimulus integer."""
    if val == 0:
        return cfg.TRIG_TARGET          # 255
    if is_test_run:
        return int(val)                 # 1..200
    cat, _ = decode_train_stim(val)
    return int(cat % 254 + 1)          # compress to 1..254


# ── Trigger helper ─────────────────────────────────────────────────────────────

def send_trigger(p_port, code):
    """Send a parallel-port trigger pulse. No-op if p_port is None."""
    if p_port is None or code is None:
        return
    p_port.setData(int(code))
    core.wait(0.002)
    p_port.setData(0)


# ── Save & quit ────────────────────────────────────────────────────────────────

def save_and_quit(sub_id, ses, run_num, run_label, results, stim_log, p_port, win):
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    base = os.path.join(
        cfg.DATA_DIR,
        f"sub-{sub_id}_ses-{ses}_run-{run_num:02d}_{run_label}"
    )
    pd.DataFrame(results).to_csv(base + '_behavior.csv',  index=False)
    pd.DataFrame(stim_log).to_csv(base + '_stimlog.csv',  index=False)
    print(f"Saved: {base}_*.csv")
    if p_port:
        p_port.setData(0)
    win.close()
    core.quit()


# ── Main ───────────────────────────────────────────────────────────────────────

def run_task():

    # ── GUI dialog ────────────────────────────────────────────────────────────
    info = {
        'Subject':  '',
        'Session':  '',
        'Run':      '',     # 1-4 = test, 5-9 = train partition 1-5
        'Debug':    False,
    }
    dlg = gui.DlgFromDict(
        info, title='THINGS-EEG',
        order=['Subject', 'Session', 'Run', 'Debug']
    )
    if not dlg.OK:
        core.quit()

    sub_id  = info['Subject'].strip()
    ses     = int(info['Session'])
    run_num = int(info['Run'])
    DEBUG   = bool(info['Debug'])

    is_test_run = run_num <= cfg.TEST_RUNS
    run_label   = 'test' if is_test_run else 'train'

    # ── Load stimulus order ───────────────────────────────────────────────────
    if is_test_run:
        order_path     = os.path.join(cfg.ORDER_DIR, f'sub-{sub_id}_ses-{ses}_test.npy')
        full_order     = np.load(order_path)                  # (20, 51, 4)
        run_order      = full_order[:, :, run_num - 1]        # (20, 51)
        seqs_per_block = cfg.TEST_SEQ_PER_BLOCK               # 51
        global_block_offset = 0

    else:
        order_path     = os.path.join(cfg.ORDER_DIR, f'sub-{sub_id}_ses-{ses}_train.npy')
        full_order     = np.load(order_path)                  # (20, 56, 15)
        block_idx      = run_num - cfg.TEST_RUNS - 1          # run 5→block 0, run 19→block 14
        run_order      = full_order[:, :, block_idx]          # (20, 56)
        seqs_per_block = cfg.TRAIN_SEQ_PER_BLOCK              # 56
        global_block_offset = block_idx

    # ── Build image lookups ───────────────────────────────────────────────────
    test_lookup, train_lookup = build_image_lookup()

    # Target images
    target_folder = os.path.join(cfg.IMG_DIR, 'target')
    if os.path.isdir(target_folder):
        target_files = sorted([
            os.path.join(target_folder, f)
            for f in os.listdir(target_folder)
            if f.lower().endswith(('.jpg', '.png', '.bmp'))
            and not f.startswith('._')
        ])
    else:
        target_files = []

    if not target_files:
        print("Warning: No target images found — blue placeholder will be used for targets.")

    # ── Window ────────────────────────────────────────────────────────────────
    if DEBUG:
        win = visual.Window(
            size=[1024, 768],
            fullscr=False,
            screen=0,
            color=cfg.BACKGROUND_COLOR,
            units='pix',
            allowGUI=True,
        )
    else:
        win = visual.Window(
            size=cfg.SCREEN_RES,
            fullscr=True,
            screen=cfg.SUBJECT_SCREEN,      # 1 = subject-side monitor
            color=cfg.BACKGROUND_COLOR,
            units='pix',
            allowGUI=False,
            waitBlanking=True,
        )

    frame_dur  = win.monitorFramePeriod
    flip_slack = frame_dur * 0.5
    win_w, win_h = win.size

    if platform.system() == 'Darwin':  # macOS Retina
        win_w, win_h = win.size[0] / 2, win.size[1] / 2
    else:
        win_w, win_h = win.size[0], win.size[1]
    print(f"Monitor: frame_dur={frame_dur*1000:.2f}ms  flip_slack={flip_slack*1000:.2f}ms")

    # ── Parallel port ─────────────────────────────────────────────────────────
    p_port = None
    if not DEBUG:
        try:
            p_port = parallel.Parallel(cfg.PARALLEL_PORT_ADDR)
            p_port.setData(0)
            print("Parallel port ready.")
        except Exception as e:
            print(f"Parallel port unavailable ({e}). Triggers disabled.")

    # ── Visual objects ────────────────────────────────────────────────────────

    # Bull's-eye fixation (matches MATLAB bullsEyeFixation_2)
    fix_outer = visual.Circle(win, radius=12, fillColor='black', lineColor=None, opacity=0.5)
    fix_inner = visual.Circle(win, radius=6,  fillColor='red',   lineColor=None, opacity=0.5)

    # Main stimulus
    img_stim    = visual.ImageStim(win, size=cfg.STIM_SIZE_PIX)
    placeholder = visual.Rect(win, width=cfg.STIM_SIZE_PIX[0],
                              height=cfg.STIM_SIZE_PIX[1],
                              fillColor='blue', lineColor=None)

    # Photodiode square — top-right corner
    pd_size = cfg.PHOTODIODE_SIZE
    pd_rect = visual.Rect(
        win,
        width=pd_size, height=pd_size,
        pos=(win_w / 2 - pd_size / 2, win_h / 2 - pd_size / 2),
        lineColor=None,
    )

    txt_stim = visual.TextStim(win, color='black', height=28, wrapWidth=800)

    # ── Helper draw functions ─────────────────────────────────────────────────

    def draw_fixation():
        fix_outer.draw()
        fix_inner.draw()

    def draw_photodiode(on):
        pd_rect.fillColor = 'white' if on else cfg.BACKGROUND_COLOR
        pd_rect.draw()

    def flip_blank(photodiode_on=False, deadline=None):
        """Draw fixation + photodiode and flip, optionally at a deadline."""
        draw_fixation()
        draw_photodiode(photodiode_on)
        return win.flip(deadline) if deadline is not None else win.flip()

    def prepare_image(val, is_test):
        """Load image into img_stim or set up placeholder. Does NOT draw."""
        try:
            if val == 0:
                if target_files:
                    img_stim.image = np.random.choice(target_files)
                    return 'image'
                else:
                    return 'placeholder'
            else:
                lookup = test_lookup if is_test else train_lookup
                img_stim.image = lookup[val]
                return 'image'
        except Exception as e:
            print(f"Image load failed for stim_val={val}: {e}")
            return 'placeholder'

    def draw_stimulus(img_type, is_target):
        """Draw whichever stimulus was prepared."""
        if img_type == 'image':
            img_stim.draw()
        else:
            placeholder.draw()
    
    # ── Experimenter info screen (shown on experimenter monitor only) ─────────
    eeg_filename = f"sub-{sub_id}_ses-{ses:02d}_run-{run_num:02d}_{run_label}"

    # Print to terminal for copy-paste
    print(f"\n{'='*50}")
    print(f"  EEG filename: {eeg_filename}")
    print(f"{'='*50}\n")

    exp_win = visual.Window(
        size=[800, 400],
        fullscr=False,
        screen=0 if cfg.SUBJECT_SCREEN == 1 else 1,                           
        color='black',
        units='pix',
        allowGUI=True,
    )
    exp_txt = visual.TextStim(
        exp_win,
        text=(
            f"EXPERIMENTER\n\n"
            f"Set EEG filename to:\n\n"
            f"  {eeg_filename}\n\n\n"
            f"Press ENTER key when recording is ready."
        ),
        color='white', height=24, wrapWidth=700
    )
    exp_txt.draw()
    exp_win.flip()
    event.waitKeys(keyList=['return'])
    exp_win.close()


    # ── Instructions ──────────────────────────────────────────────────────────
    txt_stim.text = (
        "WELCOME TO THIS EXPERIMENT\n\n\n"
        "Sequences of images will be presented to you.\n\n"
        "Your task is to detect BUZZ LIGHTYEAR.\n\n"
        "RIGHT ARROW  →  Buzz is present\n"
        "LEFT ARROW   →  Buzz is absent\n\n"
        "Be as accurate as possible.\n\n\n\n\n\n\n"
        "Press any key to begin."
    )
    txt_stim.draw()
    if target_files:
        target_preview = visual.ImageStim(
            win, image=target_files[0],
            size=[win_h * 0.2, win_h * 0.2],   # 20% of screen height
            pos=[0, -win_h * 0.2]   
        )
        target_preview.draw()
    draw_photodiode(False)
    win.flip()
    event.waitKeys()

    # ── Session-start trigger ─────────────────────────────────────────────────
    send_trigger(p_port, cfg.TRIG_START)

    # ── Result accumulators ───────────────────────────────────────────────────
    results  = []
    stim_log = []

    # ── Main loops ────────────────────────────────────────────────────────────
    for s in range(seqs_per_block):

        seq_stims = run_order[:, s].astype(int)        # (20,)
        target_in_seq = int(0 in seq_stims)

        # ── Pre-sequence fixation ─────────────────────────────────────────
        flip_blank(photodiode_on=False)
        core.wait(cfg.PRE_SEQ)
        if event.getKeys(keyList=['escape']):
            save_and_quit(sub_id, ses, run_num, run_label,
                            results, stim_log, p_port, win)

        # ── RSVP image stream ─────────────────────────────────────────────
        # Pre-load first image before entering the timed loop
        img_type  = prepare_image(seq_stims[0], is_test_run)
        t_seq_start = None

        for i in range(cfg.IMG_PER_SEQUENCE):
            val       = int(seq_stims[i])
            is_target = val == 0
            trig_code = get_trigger_code(val, is_test_run)

            # ── Draw prepared image into back buffer ──────────────────────
            draw_stimulus(img_type, is_target)
            draw_fixation()
            draw_photodiode(on=True)

            # ── Flip ON ───────────────────────────────────────────────────
            # First image: flip immediately and record sequence start time
            # Subsequent images: flip exactly SOA after previous onset
            if t_seq_start is None:
                t_on       = win.flip()
                t_seq_start = t_on
            else:
                t_on = win.flip(t_seq_start + i * cfg.SOA - flip_slack)

            # ── Trigger ───────────────────────────────────────────────────
            if p_port:
                core.wait(cfg.TRIGGER_DELAY)
                send_trigger(p_port, trig_code)

            # ── Pre-load NEXT image during the image-on period ────────────
            # This hides loading latency inside the blank ISI
            if i + 1 < cfg.IMG_PER_SEQUENCE:
                next_img_type = prepare_image(seq_stims[i + 1], is_test_run)

            # ── Wait for IMG_DUR to elapse since t_on ─────────────────────
            core.wait(cfg.IMG_DUR - (core.getTime() - t_on) - flip_slack)

            # ── Flip OFF (blank + fixation, photodiode black) ─────────────
            flip_blank(photodiode_on=False)

            # ── Escape check ──────────────────────────────────────────────
            if event.getKeys(keyList=['escape']):
                save_and_quit(sub_id, ses, run_num, run_label,
                                results, stim_log, p_port, win)

            # ── Log ───────────────────────────────────────────────────────
            stim_log.append({
                'sub':       sub_id,
                'ses':       ses,
                'run':       run_num,
                'run_type':  run_label,
                'block':     global_block_offset + 1,
                'sequence':  s + 1,
                'img_pos':   i + 1,
                'stim_val':  val,
                'is_target': int(is_target),
                'trigger':   trig_code,
                't_on':      t_on,
            })

            # Carry next image type into the next iteration
            if i + 1 < cfg.IMG_PER_SEQUENCE:
                img_type = next_img_type

        # ── Post-sequence: rest then response prompt ──────────────────────
        core.wait(cfg.POST_SEQ)
        txt_stim.text = "Blink — then respond"
        txt_stim.draw()
        draw_photodiode(False)
        win.flip()

        # ── Response collection ───────────────────────────────────────────
        resp_clock = core.Clock()
        keys = event.waitKeys(
            maxWait=cfg.RESP_WIN,
            keyList=['left', 'right', 'escape'],
            timeStamped=resp_clock,
        )

        if keys and keys[0][0] == 'escape':
            save_and_quit(sub_id, ses, run_num, run_label,
                            results, stim_log, p_port, win)

        # RIGHT = target present (1), LEFT = absent (0), no resp = 2
        if keys:
            response = 1 if keys[0][0] == 'right' else 0
        else:
            response = 2

        correct = int(response == target_in_seq) if response != 2 else -1

        results.append({
            'sub':       sub_id,
            'ses':       ses,
            'run':       run_num,
            'run_type':  run_label,
            'block':     global_block_offset + 1,
            'sequence':  s + 1,
            'target':    target_in_seq,
            'response':  response,
            'correct':   correct,
        })

    # ── End of run ────────────────────────────────────────────────────────────
    send_trigger(p_port, cfg.TRIG_STOP)
    save_and_quit(sub_id, ses, run_num, run_label, results, stim_log, p_port, win)


if __name__ == "__main__":
    run_task()
