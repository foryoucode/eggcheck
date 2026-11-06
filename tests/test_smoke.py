import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.vision import find_trays
from backend.anomaly import NormalBank

def test_empty_frame():
    frame=np.zeros((720,1280,3),dtype=np.uint8)
    assert find_trays(frame)==[]

def test_model_files_exist():
    root=Path(__file__).resolve().parents[1]
    assert (root/'models/neural_model.pt').is_file()
    assert (root/'models/normal_bank.npz').is_file()
    import hashlib
    assert hashlib.sha256((root/'simulator/index.html').read_bytes()).hexdigest()=='99071d6800bbc338e22e361d3eeea1258d8db94ed9697b2f1702e3a263caf52e'
