import numpy as np
import os


def generate_npz():
    
    dst_dir = './npz'
    os.makedirs(dst_dir, exist_ok = True)
    batch_sizes = [1, 16, 32, 64, 128]
    for b in batch_sizes:
        data = np.random.rand(*(b, 3, 256, 256)).astype(np.float32)
        filename = '%d-3-256-256.npz' % b
        filepath = os.path.join(dst_dir, filename)
        np.savez(filepath, data = data)

if __name__ == "__main__":
    generate_npz()
