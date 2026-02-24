import os

# ── Hardware ───────────────────────────────────────────────────────────────────
REFRESH_RATE         = 360          # Hz
SCREEN_RES           = [1920, 1080]
SUBJECT_SCREEN       = 1           # 0 = experimenter monitor, 1 = subject monitor
PARALLEL_PORT_ADDR   = '/dev/parport0'

# ── Timing (seconds, matching THINGS-EEG data) ──────────────────────────────────
IMG_DUR              = 0.100        # 100 ms image on screen
SOA                  = 0.200        # 200 ms stimulus onset asynchrony
PRE_SEQ              = 0.750        # 750 ms fixation before sequence onset
POST_SEQ             = 0.750        # 750 ms after last image before response prompt
RESP_WIN             = 2.000        # 2 s response window
TRIGGER_DELAY        = 0.011        # 11 ms delay before trigger (half flip + 3ms)

# ── Triggers ────────────────────────────────────────────────────────────────────
TRIG_START           = None           # EEG recording start (set to None because we actually need the 10 trigger for images)
TRIG_STOP            = None           # EEG recording stop  (set to None because we actually need the 40 trigger for images)
TRIG_TARGET          = 255            # Buzz Lightyear catch trial

# ── Display ────────────────────────────────────────────────────────────────────
BACKGROUND_COLOR     = [0.247, 0.271, 0.298]  # Grey
STIM_SIZE_PIX        = [500, 500]             # stimulus size in pixels
PHOTODIODE_SIZE      = 50                     # white square side length in pixels 

# ── Presentation structure ─────────────────────────────────────────────────────
IMG_PER_SEQUENCE     = 20
TRAIN_SEQ_PER_BLOCK  = 56
TEST_SEQ_PER_BLOCK   = 51
TRAIN_BLOCKS         = 15           
TEST_RUNS            = 4            
TARGETS_PER_BLOCK    = 6

# ── Stimuli ────────────────────────────────────────────────────────────────────
TOT_TRAINING_CAT     = 1654
TRAINING_CAT         = TOT_TRAINING_CAT // 2  # 827 per session
IMG_PER_CAT          = 10
REP_TRAINING         = 2

TEST_CAT             = 200
IMG_PER_TEST_CAT     = 1
REP_TEST             = 20

# ── Folders ────────────────────────────────────────────────────────────────────
ORDER_DIR            = 'stimuli_orders'
DATA_DIR             = 'data'
IMG_DIR              = 'stimuli'
