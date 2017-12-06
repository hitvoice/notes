import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=5, metadata=metadata)

fig = plt.figure(figsize=(12, 8))

bins = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(float)
centroids = (bins[1:] + bins[:-1]) / 2

counts = np.ones(8, dtype=float) * 100
width = 0.9 * (bins[1] - bins[0])
font = {'family': 'serif',
        'color': 'navy',
        'weight': 'bold',
        'size': 20,
        }

with writer.saving(fig, "writer_test.mp4", 100):  # don't change this 100
    from tqdm import tqdm
    for i in tqdm(range(100)):
        plt.clf()
        plt.xlim(min(bins), max(bins))
        plt.ylim(50, 150)
        plt.gca().set_xticks(bins)
        plt.text(1, 140, f'iter={i + 1}', fontdict=font)
        counts += np.random.randint(-3, 4, size=8)
        plt.bar(centroids, counts, align='center', width=width, color='darkolivegreen')
        writer.grab_frame()
