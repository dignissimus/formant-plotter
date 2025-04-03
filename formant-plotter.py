import seaborn as sns
import numpy as np
import scipy
import matplotlib.pyplot as plt
from librosa import lpc
import matplotlib.animation as animation
import argparse
import os
sns.set_theme()

parser = argparse.ArgumentParser(
    prog="formant-plotter",
    description="Plots formants"
)
parser.add_argument("file_name", help="Input wav file name")
parser.add_argument("--save", help="Save to formants.mp4", action="store_true")
args = parser.parse_args()

if not os.path.exists(args.file_name):
    print(f"`{args.file_name}` does not exist")
    exit(1)

rate, data = scipy.io.wavfile.read(args.file_name)
print("Audio shape", data.shape)
if len(data.shape) > 1:
    data = data[:, 1]

# Assume mono
assert len(data.shape) == 1

p = 50

# My normalise function doesn't do anything
def normalise(xs):
    return xs

formants = []

# 20ms windows
window_size = rate // 50

median = np.quantile(np.abs(data), .5)
for start in range(0, len(data), window_size):
    window = np.float64(data[start : start + window_size])
    window = normalise(window)

    # discard silence
    if max(window) < median:
        formants.append([])
        continue

    A = lpc(window, order=p)
    roots = np.sort(np.roots(A))
    roots = roots[np.imag(roots) > 0]
    bandwidth = -1/2 * rate / (2 * np.pi) * np.log(np.abs(roots))
    roots = roots[bandwidth < 400]
    angles = np.angle(roots)
    fs =  sorted(angles / (2 * np.pi) * rate)[:2]
    formants.append(fs)

frames = formants
F1 = [f[0] if f else np.nan for f in frames]
F2 = [f[1] if f else np.nan for f in frames]

fig, ax = plt.subplots()
ax.set_xlim(2500, 500)
ax.set_ylim(900, 200)
ax.set_xlabel("F2")
ax.set_ylabel("F1")
ax.set_title("Formant Tracking")
line1, = ax.plot([], [], 'ro-', label='Estimated tongue position')
ax.legend()

def update(frame):
    line1.set_data([F2[frame]], [F1[frame]])
    return line1,

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=20, blit=True)
if args.save:
    ani.save("formants.mp4")
plt.show()
