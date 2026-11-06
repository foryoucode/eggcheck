"""Pixel-only quality inspector HTTP API.

The backend receives JPEG/PNG bytes captured from a browser tab.
It has no DOM/browser/simulator controls and does not read labels or JS.
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path
from threading import Lock
from typing import Annotated

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .vision import find_trays, crop_tray
from .anomaly import NormalBank
from .neural import DefectClassifier

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / 'models'
MAX_BYTES = 2_500_000
MAX_DIM = 1600

app = FastAPI(title='EggCheck Vision', docs_url=None, redoc_url=None)
model = None
bank = None
feature_mean = None
feature_std = None
model_lock = Lock()


def load_model():
    global model, bank, feature_mean, feature_std
    ckpt = torch.load(WEIGHTS/'neural_model.pt', map_location='cpu', weights_only=True)
    net = DefectClassifier()
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    net = net.to('cpu')
    model = net
    saved = np.load(WEIGHTS/'normal_bank.npz', allow_pickle=False)
    bank = NormalBank(saved=saved)
    feature_mean = np.asarray(ckpt['feature_mean'], dtype=np.float32)
    feature_std = np.asarray(ckpt['feature_std'], dtype=np.float32)


@app.on_event('startup')
def startup():
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    load_model()


@app.get('/api/health')
def health():
    return {'ok':model is not None, 'model':'trained-normal-bank-mlp', 'pixel_only':True}


@app.post('/api/analyze')
async def analyze(req: Request):
    # No HTML payloads, metadata or simulator settings are accepted.
    size = req.headers.get('content-length')
    if size and (not size.isdigit() or int(size)>MAX_BYTES):
        raise HTTPException(413, 'Image is too large')
    raw = await req.body()
    if len(raw)>MAX_BYTES:
        raise HTTPException(413,'Image is too large')
    if not raw:
        raise HTTPException(400,'Missing image')
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8),cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400,'Expected JPEG or PNG image')
    h,w=img.shape[:2]
    if max(w,h)>MAX_DIM:
        raise HTTPException(400,'Frame must be <= 1600 px on each side')
    if w<150 or h<150:
        raise HTTPException(400,'Frame is too small')
    # Bound CPU consumption. Each crop is compared with the learned visual reference bank.
    with model_lock:
        start=time.perf_counter()
        boxes=find_trays(img)[:16]
        results=[]
        if boxes:
            features=np.stack([bank.get_features(crop_tray(img,b)) for b in boxes]).astype(np.float32)
            features=(features-feature_mean)/feature_std
            with torch.inference_mode():
                probs=torch.softmax(model(torch.from_numpy(features)),dim=1)[:,1].numpy()
            for (x,y,bw,bh),p in zip(boxes,probs):
                results.append({'x':x,'y':y,'w':bw,'h':bh,'defect_probability':round(float(p),4)})
        duration_ms=round((time.perf_counter()-start)*1000,1)
    return {'width':w,'height':h,'detections':results,'latency_ms':duration_ms, 'model':'normal-bank-mlp'}


app.mount('/simulator',StaticFiles(directory=ROOT/'simulator',html=True),name='simulator')
app.mount('/assets',StaticFiles(directory=ROOT/'frontend'),name='assets')

@app.get('/')
def homepage():
    return FileResponse(ROOT/'frontend'/'index.html',media_type='text/html')
