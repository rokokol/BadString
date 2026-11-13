import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image, ImageDraw

from badstring import sine_basis
from growing_circle import IMPULSE_STRENGTH, advance

WIDTH = 201
HEIGHT = 151
UPDATE_RATE = 60
FPS = 60
SPRING_STIFFNESS = 25000.0
DAMPING = 70.0
EXPORT_WIDTH = 800
EXPORT_HEIGHT = 600


def boundaries(mask):
    height, width = mask.shape
    transitions = mask[1:] != mask[:-1]
    present = transitions.any(axis=0)
    upper = np.zeros(width)
    lower = np.zeros(width)
    center = (height - 1) / 2
    upper[present] = np.argmax(transitions[:, present], axis=0) + 0.5 - center
    lower[present] = (
        height - 1.5 - np.argmax(transitions[::-1, present], axis=0) - center
    )
    background = np.count_nonzero(mask) > mask.size / 2
    upper[mask[0] != background] = -center
    lower[mask[-1] != background] = center
    upper[[0, -1]] = 0
    lower[[0, -1]] = 0
    return np.stack((upper, lower))


def read_video(path):
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={UPDATE_RATE},scale={WIDTH}:{HEIGHT}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    size = WIDTH * HEIGHT
    if not result.stdout or len(result.stdout) % size:
        raise ValueError("Video did not produce complete frames")
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH)
    return frames < 128


def damped_advance(position, velocity, target, dt, stiffness, damping):
    modes = np.arange(1, len(position) + 1)
    omega2 = (np.pi * modes) ** 2 + stiffness
    equilibrium = stiffness * target / omega2
    relative = position - equilibrium
    half_damping = damping / 2
    root = np.sqrt((half_damping**2 - omega2).astype(complex))
    exp_plus = np.exp((-half_damping + root) * dt)
    exp_minus = np.exp((-half_damping - root) * dt)
    cosine = (exp_plus + exp_minus) / 2
    sine = np.empty_like(root)
    critical = np.abs(root) < 1e-10
    sine[critical] = dt * np.exp(-half_damping * dt)
    sine[~critical] = (exp_plus[~critical] - exp_minus[~critical]) / (
        2 * root[~critical]
    )
    next_position = (
        equilibrium + relative * cosine + (velocity + half_damping * relative) * sine
    )
    next_velocity = (
        velocity * cosine - (omega2 * relative + half_damping * velocity) * sine
    )
    return next_position.real, next_velocity.real


def simulate(masks, fps=FPS, update_rate=UPDATE_RATE, mode="impulse"):
    if mode not in ("impulse", "spring"):
        raise ValueError(f"Unknown mode: {mode}")
    if fps % update_rate:
        raise ValueError("The frame rate must be divisible by the update rate")
    width = masks.shape[2]
    basis = sine_basis(width)

    def coefficients(mask):
        return 2 / (width - 1) * (boundaries(mask)[:, 1:-1] @ basis)

    position = coefficients(masks[0])
    velocity = np.zeros_like(position)
    target = position.copy()
    period = fps // update_rate
    result = np.zeros((len(masks) * period + 1, 2, width))
    result[0, :, 1:-1] = position @ basis
    for frame in range(1, len(result)):
        for i in range(2):
            if mode == "spring":
                position[i], velocity[i] = damped_advance(
                    position[i],
                    velocity[i],
                    target[i],
                    1 / fps,
                    SPRING_STIFFNESS,
                    DAMPING,
                )
            else:
                position[i], velocity[i] = advance(
                    position[i], velocity[i], target[i], 1 / fps, 0
                )
        if frame % period == 0 and frame // period < len(masks):
            target = coefficients(masks[frame // period])
            if mode == "impulse":
                velocity += IMPULSE_STRENGTH * (target - position)
        result[frame, :, 1:-1] = position @ basis
    return result


def render_frame(strings):
    image = Image.new("RGB", (EXPORT_WIDTH, EXPORT_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    x = np.linspace(0, EXPORT_WIDTH - 1, strings.shape[1])
    for string in strings:
        y = (string + (HEIGHT - 1) / 2) * (EXPORT_HEIGHT - 1) / (HEIGHT - 1)
        draw.line(list(zip(x.tolist(), y.tolist())), fill="black", width=3)
    return image


def export_video(frames, source, output):
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{EXPORT_WIDTH}x{EXPORT_HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-shortest",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            process.stdin.write(render_frame(frame).tobytes())
        process.stdin.close()
    except BrokenPipeError:
        process.stdin.close()
        process.wait()
        raise RuntimeError("FFmpeg stopped before all frames were written")
    if process.wait() != 0:
        raise RuntimeError("FFmpeg failed to export the video")


def main():
    parser = argparse.ArgumentParser(description="Animate two strings from a video")
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("badapple-source.mp4"),
    )
    parser.add_argument("--mode", choices=("spring", "impulse"), default="spring")
    parser.add_argument(
        "--export", type=Path, help="Write an MP4 with the source audio"
    )
    args = parser.parse_args()
    frames = simulate(read_video(args.video), mode=args.mode)
    if args.export:
        export_video(frames, args.video, args.export)
        return
    fig, ax = plt.subplots()
    x = np.arange(WIDTH)
    (upper,) = ax.plot(x, frames[0, 0], color="black")
    (lower,) = ax.plot(x, frames[0, 1], color="black")
    ax.set_xlim(0, WIDTH - 1)
    ax.set_ylim((HEIGHT - 1) / 2, -(HEIGHT - 1) / 2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()

    def update(frame):
        upper.set_ydata(frames[frame, 0])
        lower.set_ydata(frames[frame, 1])
        return upper, lower

    animation = FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / FPS, repeat=False
    )
    plt.show()
    return animation


if __name__ == "__main__":
    main()
