import argparse
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image


SPEEDS = (0.5, 1.0, 2.0, 4.0)


@lru_cache(maxsize=8)
def sine_basis(size):
    positions = np.arange(1, size - 1)
    return np.sin(np.pi * np.outer(positions, positions) / (size - 1))


def contours(mask):
    height, width = mask.shape
    if height < 2 or width < 3:
        raise ValueError("The image must be at least 3 pixels wide and 2 pixels high")
    center = (height - 1) / 2
    present = mask.any(axis=0)
    top = np.zeros(width)
    bottom = np.zeros(width)
    top[present] = np.argmax(mask[:, present], axis=0) - center
    bottom[present] = height - 1 - np.argmax(mask[::-1, present], axis=0) - center
    top[[0, -1]] = 0
    bottom[[0, -1]] = 0
    return top, bottom


def evolve(initial, time, speed):
    basis = sine_basis(len(initial))
    coefficients = 2 / (len(initial) - 1) * (basis @ initial[1:-1])
    modes = np.arange(1, len(initial) - 1)
    result = np.zeros_like(initial, dtype=float)
    result[1:-1] = basis @ (coefficients * np.cos(np.pi * modes * speed * time))
    return result


def main():
    parser = argparse.ArgumentParser(description="Animate two strings from a black and white image")
    parser.add_argument("image", nargs="?", help="Image path; omit it to use a black circle")
    args = parser.parse_args()

    if args.image:
        image = Image.open(args.image).convert("L")
        image.thumbnail((500, 500))
        mask = np.asarray(image) < 128
    else:
        y, x = np.ogrid[:301, :501]
        mask = (x - 250) ** 2 + (y - 150) ** 2 <= 75 ** 2

    top, bottom = contours(mask)
    height, width = mask.shape
    center = (height - 1) / 2
    duration = 2.0
    fps = 60
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    x = np.arange(width)
    lines = []
    for ax, speed in zip(axes.flat, SPEEDS):
        ax.imshow(~mask, cmap="gray", vmin=0, vmax=1, alpha=0.2, extent=(0, width - 1, height - 1, 0))
        upper, = ax.plot(x, center + top, color="tab:blue")
        lower, = ax.plot(x, center + bottom, color="tab:red")
        ax.set_xlim(0, width - 1)
        ax.set_ylim(height - 1, 0)
        ax.set_aspect("equal")
        ax.set_title(f"a = {speed:g}")
        lines.append((upper, lower))
    title = fig.suptitle("t = 0.00 s")

    def update(frame):
        time = frame / fps
        for speed, (upper, lower) in zip(SPEEDS, lines):
            upper.set_ydata(center + evolve(top, time, speed))
            lower.set_ydata(center + evolve(bottom, time, speed))
        title.set_text(f"t = {time:.2f} s")
        return tuple(line for pair in lines for line in pair)

    animation = FuncAnimation(fig, update, frames=range(int(duration * fps) + 1),
                              interval=1000 / fps, repeat=False, blit=False)
    plt.show()
    return animation


if __name__ == "__main__":
    main()
