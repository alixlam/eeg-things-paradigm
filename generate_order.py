import numpy as np
import random, os
import config as cfg

def get_constrained_pool(pool, size):
    """Fills a sequence while preventing AA and ABA patterns."""
    seq = []
    attempts = 0
    while len(seq) < size:
        item = random.choice(pool)
        # Check for AA (immediate) and ABA (one gap)
        if len(seq) >= 1 and item == seq[-1]:
            continue
        if len(seq) >= 2 and item == seq[-2]:
            continue
        
        seq.append(item)
        pool.remove(item)
    return seq

def create_orders():
    sub_id = input("Enter Subject ID: ")
    
    # 200 Test images, repeated 80 times across study
    all_test = [(999, i) for i in range(1, 201)] * 80
    random.shuffle(all_test)
    
    # 1654 Train categories
    all_train_cats = np.random.permutation(np.arange(1, 1655))
    
    for ses in range(1, 5):
        session_plan = {}
        start_idx = 0 if ses in [1, 3] else 827
        ses_cats = all_train_cats[start_idx : start_idx + 827]
        # 10 images per cat, 2 reps each as per THINGS-EEG design
        ses_train_pool = [(c, i) for c in ses_cats for i in range(1, 11)] * 2
        random.shuffle(ses_train_pool)

        for run in range(1, 13):
            is_test = run <= 4
            n_seqs = 51 if is_test else 56
            grid = np.zeros((20, n_seqs), dtype=object)
            target_seqs = random.sample(range(n_seqs), 6)
            
            for s in range(n_seqs):
                # Calculate how many non-target images we need
                n_needed = 19 if s in target_seqs else 20
                
                # Get constrained images from the appropriate pool
                pool = all_test if is_test else ses_train_pool
                constrained_images = get_constrained_pool(pool, n_needed)
                
                t_pos = random.randint(1, 19) if s in target_seqs else -1
                img_idx = 0
                for i in range(20):
                    if i == t_pos:
                        grid[i, s] = (0, 0) # Target
                    else:
                        grid[i, s] = constrained_images[img_idx]
                        img_idx += 1
            session_plan[run] = grid
        
        np.save(os.path.join(cfg.ORDER_DIR, f'sub-{sub_id}_ses-{ses}_master.npy'), session_plan)
    print("Orders generated successfully without AA/ABA repeats.")

if __name__ == "__main__":
    create_orders()