"""Only raw image processing. Never imports / communicates with the HTML demo."""
from __future__ import annotations
import cv2
import numpy as np

IMAGE_SIZE = 128

def find_trays(bgr: np.ndarray):
    """Detect complete, approximately rectangular bright trays on a dark belt.

    The step finds OBJECT locations only. The CNN, not these thresholds,
    decides whether each object is defective.
    """
    if bgr is None or bgr.size == 0:
        return []
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 88).astype(np.uint8) * 255
    kw = max(3, int(min(h, w) * .018) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw,kw))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,bw,bh = cv2.boundingRect(c)
        ratio = bw / max(bh, 1)
        rect_area = bw * bh
        if not (.74 <= ratio <= 1.60):
            continue
        if bw < max(45, w * .075) or bh < max(45, h * .18):
            continue
        if rect_area < w*h*.009 or cv2.contourArea(c) < rect_area*.16:
            continue
        # Fully visible tray: cropped partial trays must not receive unreliable labels.
        if x <= 2 or y <= 2 or x+bw >= w-2 or y+bh >= h-2:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))
    return sorted(boxes, key=lambda b:(b[0],b[1]))

def crop_tray(bgr: np.ndarray, box):
    h,w=bgr.shape[:2]
    x,y,bw,bh=box
    margin=max(2,int(min(bw,bh)*.018))
    x0=max(0,x-margin);y0=max(0,y-margin)
    x1=min(w,x+bw+margin);y1=min(h,y+bh+margin)
    return bgr[y0:y1,x0:x1].copy()

def prepare_gray(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gray = cv2.resize(gray,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_AREA)
    # Gray-only input discourages shortcut learning by tray color.
    return gray.astype(np.float32) / 255.


def preprocess_torch(bgr):
    return prepare_gray(bgr)[None, :, :]
