"""Turn a clip into a .y4m file Chrome can use as its webcam.

    python -m tests.fake_camera data/raw/bowling/07391.mp4 out.y4m [repeats]

Chrome: --use-fake-device-for-media-stream --use-file-for-fake-video-capture=out.y4m
Each repeat is preceded by 1 s of the first frame held still, like a signer
pausing between signs. Chrome loops the file.
"""
import sys

import cv2


def write_y4m(src: str, dst: str, repeats: int = 1) -> int:
    cap = cv2.VideoCapture(src)
    fps = round(cap.get(cv2.CAP_PROP_FPS)) or 25
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        frames.append(cv2.resize(f, (640, int(h * 640 / w) // 2 * 2)))  # 4:2:0 needs even dims
    cap.release()
    if not frames:
        raise SystemExit(f"could not read {src}")
    h, w = frames[0].shape[:2]
    sequence = ([frames[0]] * fps + frames) * repeats
    with open(dst, "wb") as out:
        out.write(f"YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C420jpeg\n".encode())
        for f in sequence:
            out.write(b"FRAME\n" + cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420).tobytes())
    return len(sequence)


if __name__ == "__main__":
    n = write_y4m(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    print(f"wrote {sys.argv[2]} ({n} frames)")
