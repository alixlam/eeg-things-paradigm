from psychopy import visual, core, event, gui, parallel
import numpy as np
import pandas as pd
import config as cfg
import os

def get_image_path(cat, img):
    if cat == 0: return os.path.join(cfg.IMG_DIR, 'target', 'target.jpg')
    if cat == 999: return os.path.join(cfg.IMG_DIR, 'test', f"{img}.jpg")
    return os.path.join(cfg.IMG_DIR, 'training', f"cat_{str(cat).zfill(4)}", f"{img}.jpg")

def run_task():
    info = {'Subject': '1', 'Session': '1', 'Run': range(1, 13), 'Debug': False}
    if not gui.DlgFromDict(info).OK: core.quit()
    
    DEBUG = info['Debug']
    win = visual.Window(size=cfg.SCREEN_RES if not DEBUG else [800, 600], 
                        fullscr=not DEBUG, color=[-1,-1,-1], units='pix')
    
    p_port = None
    if not DEBUG:
        try:
            p_port = parallel.Parallel(cfg.PARALLEL_PORT_ADDR)
        except:
            print("Parallel Port Error: Triggering disabled.")

    # Load Order
    path = os.path.join(cfg.ORDER_DIR, f"sub-{info['Subject']}_ses-{info['Session']}_master.npy")
    master_data = np.load(path, allow_pickle=True).item()
    run_grid = master_data[int(info['Run'])]
    
    # Stimuli
    fix = visual.Circle(win, radius=5, fillColor='red', lineColor='red')
    img_stim = visual.ImageStim(win, size=(400, 400))
    pd_rect = visual.Rect(win, size=50, pos=(cfg.SCREEN_RES[0]/2-25, cfg.SCREEN_RES[1]/2-25))

    if p_port: p_port.setData(cfg.TRIG_START); core.wait(0.01); p_port.setData(0)

    results = []
    fps = cfg.REFRESH_RATE if not DEBUG else 60
    
    for s in range(run_grid.shape[1]):
        fix.draw(); win.flip(); core.wait(cfg.PRE_SEQ)
        target_occurrence = 1 if any(v == (0,0) for v in run_grid[:, s]) else 2
        
        for i in range(20):
            cat, img = run_grid[i, s]
            is_target = (cat == 0)

            # Preparation
            if not DEBUG:
                img_stim.image = get_image_path(cat, img)
                img_stim.draw()
            else:
                visual.Rect(win, size=400, fillColor='blue' if is_target else 'grey').draw()
            
            fix.draw()
            pd_rect.fillColor = 'white'
            pd_rect.draw()
            
            # --- ON ---
            win.flip()
            if p_port:
                p_port.setData(cfg.TRIG_TARGET if is_target else (cat % 254 + 1))
                core.wait(0.002); p_port.setData(0)
            core.wait(cfg.IMG_DUR - (1.0/fps))

            # --- OFF ---
            pd_rect.fillColor = 'black'
            pd_rect.draw(); fix.draw(); win.flip()
            core.wait(cfg.SOA - cfg.IMG_DUR - (1.0/fps))

        # Response
        visual.TextStim(win, text="?").draw(); win.flip()
        keys = event.waitKeys(maxWait=cfg.RESP_WIN, keyList=['left', 'right', 'escape'])
        if 'escape' in (keys or []): core.quit()
        
        resp = 1 if 'left' in (keys or []) else 2 if 'right' in (keys or []) else 0
        results.append({'Run': info['Run'], 'Seq': s, 'Target': target_occurrence, 
                        'Resp': resp, 'Correct': 1 if resp == target_occurrence else 0})

    if p_port: p_port.setData(cfg.TRIG_STOP); core.wait(0.01); p_port.setData(0)
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg.DATA_DIR, f"sub-{info['Subject']}_ses-{info['Session']}_run-{info['Run']}.csv"), index=False)
    win.close(); core.quit()

if __name__ == "__main__":
    run_task()