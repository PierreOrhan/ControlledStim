import os.path
from typing import Union
import pathlib
import numpy as np
import librosa
import zarr as zr
from sounds.api import SoundGenerator
import soundfile as sf

class Sequence:
    name: str
    sounds: list[str]
    isi: float # in seconds

class Sound:
    name: str
    sound: np.ndarray
    samplerate: int

class Sound_pool:
    name: str
    sounds: dict[Sound]

class Combinator:
    name: str

class Protocol:
    name: str
    sequence_isi: float