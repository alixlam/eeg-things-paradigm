# THINGS-EEG Stimulus Presentation Paradigm

Python/PsychoPy reimplementation of the RSVP paradigm from:

> Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022).  
> **A large and rich EEG dataset for modeling human visual object recognition.**  
> *NeuroImage*, 264, 119754. https://doi.org/10.1016/j.neuroimage.2022.119754

---

## Paradigm Overview

Participants view rapid serial visual presentation (RSVP) sequences of 20 images and detect a target image (Buzz Lightyear).

![Paradigm (from Xu et al. 2025 [1])](paradigm_figure.png)

```
┌─────────────────────────────────────────────────────────────────┐
│                        ONE SEQUENCE                             │
│                                                                 │
│  750ms          20 × (100ms ON + 100ms OFF)          750ms      │
│  fixation  ──►  [img][blank][img][blank]...[img]  ──► fixation  │
│                                                                 │
│                         then: up to 2s response window          │
│                         (blink + key press)                     │
└─────────────────────────────────────────────────────────────────┘

RIGHT ARROW = target present   |   LEFT ARROW = target absent
```

### Session structure

Each subject completes **4 sessions**, each with **19 runs** (~5 min each):

| Runs  | Type     | Sequences/run | Images shown                          |
|-------|----------|---------------|---------------------------------------|
| 1–4   | Test     | 51            | 200 test images × 20 reps/session     |
| 5–19  | Train    | 56            | 8270 training images × 2 reps/session |

- **Test images**: 200 object concepts (1 image each), same across all 4 sessions → **80 reps per image** across the full experiment
- **Training images**: 1654 object concepts × 10 images = 16540 conditions, split into 2 halves across sessions → **4 reps per image** across the full experiment
- **Targets**: 6 per block (Buzz Lightyear), placed at random positions within sequences

---

## Files

```
├── config.py             # All parameters (timing, triggers, paths, screen)
├── generate_orders.py    # Generate randomised stimulus orders (.npy files) of images for each session/run
├── main.py               # PsychoPy task — run this for data collection
├── sanity_check.py       # Verify order files are correct before testing
└── README.md             # This file
```

---

## Setup

### 1. Install dependencies

```bash
pip install psychopy==2023.2.3
pip install .
```

### 2. Organise stimuli

```
stimuli/
├── test_images/
│   ├── 00001_aircraft_carrier/
│   │   └── aircraft_carrier_06s.jpg
│   ├── 00002_antelope/
│   └── ...  (200 folders, 1 image each)
├── training_images/
│   ├── 00001_aardvark/
│   │   ├── aardvark_01b.jpg
│   │   ├── aardvark_02s.jpg
│   │   └── ...  (10 images per folder)
│   └── ...  (1654 folders)
└── target/
    └── buzz_lightyear.jpg
```

### 3. Edit `config.py`

Set paths and hardware parameters for your setup:

```python
IMG_DIR        = 'stimuli'          # path to stimuli folder
ORDER_DIR      = 'stimuli_orders'   # where .npy order files are saved
DATA_DIR       = 'data'             # where behavioral CSVs are saved
SUBJECT_SCREEN = 1                  # index of subject monitor (0=experimenter)
PARALLEL_PORT_ADDR = '/dev/parport0'  # Linux parallel port address
```

### 4. Generate stimulus orders

Run once per subject before the experiment:

```bash
python generate_orders.py
# Enter subject ID when prompted (e.g. 01)
# Generates 8 files: sub-01_ses-{1..4}_{train,test}.npy
```

---

## Running the experiment

```bash
python main.py
```

A dialog box will appear asking for:

| Field   | Description                                      |
|---------|--------------------------------------------------|
| Subject | Subject ID (e.g. `01`)                           |
| Session | Session number (1–4)                             |
| Run     | Run number: 1–4 = test, 5–19 = train block 1–15 |
| Debug   | `True` = windowed, no parallel port              |

### Run order per session

```
Run 1  → test block 1   (51 sequences)
Run 2  → test block 2   (51 sequences)
Run 3  → test block 3   (51 sequences)
Run 4  → test block 4   (51 sequences)
Run 5  → train block 1  (56 sequences)
Run 6  → train block 2  (56 sequences)
...
Run 19 → train block 15 (56 sequences)
```

### EEG recording

Before each run the experimenter screen shows the EEG filename to use :

```
sub-01_ses-01_run-05_train
```

Start the eego recording manually, then press **Enter** to proceed to participant instructions.

---

## Output files

For each run, two CSV files are saved in `data/`:

**`*_behavior.csv`** — one row per sequence:

| Column   | Description                              |
|----------|------------------------------------------|
| sub      | Subject ID                               |
| ses      | Session number                           |
| run      | Run number                               |
| block    | Absolute block number within session     |
| sequence | Sequence number within block             |
| target   | Whether target was present (0/1)         |
| response | Right=1 (present), Left=0 (absent), 2=NR |
| correct  | 1=correct, 0=incorrect, -1=no response   |

**`*_stimlog.csv`** — one row per image presentation:

| Column    | Description                                      |
|-----------|--------------------------------------------------|
| block     | Absolute block number                            |
| sequence  | Sequence number within block                     |
| img_pos   | Position within sequence (1–20)                  |
| stim_val  | Stimulus integer (encodes image identity)        |
| is_target | 1 if catch trial                                 |
| trigger   | EEG trigger code sent via parallel port          |
| t_on      | Timestamp of image onset (PsychoPy clock)        |

### Stimulus encoding

- **Test**: `stim_val` = image index (1–200), maps directly to `test_images/000XX_*/`
- **Train**: `stim_val` = `cat * 10 - 10 + img_num`, where `cat` = folder index (1–1654), `img_num` = image within folder (1–10)
- **Target**: `stim_val` = 0, `trigger` = 255

### EEG triggers

| Code  | Meaning                        |
|-------|--------------------------------|
| 1–200 | Test image (image index)       |
| 1–254 | Train image (`cat % 254 + 1`)  |
| 255   | Target (Buzz Lightyear)        |

---

## Debug mode

Set `Debug = True` in the dialog to run on any monitor without a parallel port. Images load normally — only the window size (1024×768) and absence of EEG triggers differ from the real task.

---

## References

Xu, J., Nunes, U. B., Jiang, W., Ryther, S., Pringle, J., Scotti, P. S., ... & Kneeland, R. (2025). Alljoined-1.6 M: A Million-Trial EEG-Image Dataset for Evaluating Affordable Brain-Computer Interfaces.