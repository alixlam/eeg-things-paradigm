import os

# Hardware
REFRESH_RATE = 360  
SCREEN_RES = [1920, 1080]
PARALLEL_PORT_ADDR = '/dev/parport0' 

# Timing (Seconds)
IMG_DUR = 0.100  
SOA = 0.200      
PRE_SEQ = 0.750
POST_SEQ = 0.750
RESP_WIN = 2.0

# Triggers
TRIG_START = 10
TRIG_STOP = 40
TRIG_TARGET = 255

# Folders
ORDER_DIR = 'stimuli_orders'
DATA_DIR = 'data'
IMG_DIR = 'stimuli'