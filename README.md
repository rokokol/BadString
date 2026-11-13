# BadString

[![assets](https://img.shields.io/badge/docs-assets-555?style=flat)](ASSETS.md)

BadString turns a black-and-white video into an animation of two strings that vibrate. It includes renders of Bad Apple!! and small string experiments

## Requirements

Use Python, FFmpeg, NumPy, Matplotlib and Pillow

```bash
python -m pip install numpy matplotlib pillow
```

## Run

Provide a video file with an audio track for export. The original source video is not included in the repository

```bash
python bad_apple.py path/to/video.mp4
python bad_apple.py path/to/video.mp4 --export output.mp4
```

Run the circle experiments with `python circle_reset.py`, `python circle_spring.py` or `python circle_impulse.py`

## Tests

```bash
python -m unittest -v
```
