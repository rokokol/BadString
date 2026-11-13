import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from badstring import sine_basis


WIDTH = 201
HEIGHT = 121
DURATION = 3
FPS = 60
MAX_RADIUS = 45
SPRING_STIFFNESS = 900.0
IMPULSE_STRENGTH = 15.0


def circle_profile(time):
    radius = MAX_RADIUS * np.clip(time / DURATION, 0, 1)
    x = np.arange(WIDTH) - (WIDTH - 1) / 2
    profile = np.sqrt(np.maximum(radius * radius - x * x, 0))
    profile[[0, -1]] = 0
    return profile


def advance(position, velocity, target, dt, stiffness):
    modes = np.arange(1, len(position) + 1)
    omega = np.sqrt((np.pi * modes) ** 2 + stiffness)
    equilibrium = stiffness * target / omega**2
    phase = omega * dt
    relative = position - equilibrium
    return (equilibrium + relative * np.cos(phase) + velocity * np.sin(phase) / omega,
            -relative * omega * np.sin(phase) + velocity * np.cos(phase))


def simulate(mode):
    if mode not in ("reset", "spring", "impulse"):
        raise ValueError(f"Unknown mode: {mode}")
    basis = sine_basis(WIDTH)

    def coefficients(profile):
        return 2 / (WIDTH - 1) * (basis @ profile[1:-1])

    position = coefficients(circle_profile(0))
    velocity = np.zeros_like(position)
    target = position.copy()
    frames = np.zeros((DURATION * FPS + 1, WIDTH))
    period = 18 if mode == "reset" else 6
    for frame in range(1, len(frames)):
        position, velocity = advance(position, velocity, target, 1 / FPS,
                                     SPRING_STIFFNESS if mode == "spring" else 0)
        if frame % period == 0:
            target = coefficients(circle_profile(frame / FPS))
            if mode == "reset":
                position = target.copy()
                velocity.fill(0)
            elif mode == "impulse":
                velocity += IMPULSE_STRENGTH * (target - position)
        frames[frame, 1:-1] = basis @ position
    return frames


def run(mode):
    frames = simulate(mode)
    center_x = (WIDTH - 1) / 2
    center_y = (HEIGHT - 1) / 2
    fig, ax = plt.subplots()
    circle = plt.Circle((center_x, center_y), 0, color="black", alpha=0.18)
    ax.add_patch(circle)
    x = np.arange(WIDTH)
    upper, = ax.plot(x, center_y - frames[0], color="tab:blue")
    lower, = ax.plot(x, center_y + frames[0], color="tab:red")
    ax.set_xlim(0, WIDTH - 1)
    ax.set_ylim(HEIGHT - 1, 0)
    ax.set_aspect("equal")

    def update(frame):
        time = frame / FPS
        circle.set_radius(MAX_RADIUS * time / DURATION)
        upper.set_ydata(center_y - frames[frame])
        lower.set_ydata(center_y + frames[frame])
        ax.set_title(f"{mode}, a = 1, t = {time:.2f} s")
        return upper, lower, circle

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, repeat=False)
    plt.show()
    return animation
