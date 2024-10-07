from sounds.perExperiment.sound_elements.sound_class import Sound
import numpy as np
import librosa
from  dataclasses import dataclass,field
from typing import Union

def bip_randomPitch(samplerate: int,duration: float,fs: Union[list[float],np.ndarray]):
    dis = np.random.uniform(0, 1, len(fs))
    # weights for random linear combination of pure tones...
    return np.transpose(np.transpose(
        np.stack([librosa.tone(f, sr=samplerate, duration=duration) for f in fs],
                 axis=0)) @ np.transpose(dis))
@dataclass
class Bip_randPitch(Sound):
    name : str = "Bip_randomPitch"
    fs : Union[list[float],np.ndarray] = field(default=list) # frequencies of the pure tones.
    def __post_init__(self) -> None:
        self.sound = bip_randomPitch(self.samplerate,self.duration,self.fs)

def bip(samplerate: int,duration: float,fs: Union[list[float],np.ndarray]):
    return np.sum(np.stack([librosa.tone(f, sr=samplerate, duration=duration) for f in fs],
                 axis=0),axis=0)
@dataclass
class Bip(Sound):
    name : str = "Bip"
    fs : Union[list[float],np.ndarray] = field(default=list) # frequencies of the pure tones.
    first_freq : int = field(init=False)
    def __post_init__(self) -> None:
        self.sound = bip(self.samplerate,self.duration,self.fs)
        self.first_freq = self.fs[0]

@dataclass
class Silence(Sound):
    name : str = "Silence"
    def __post_init__(self) -> None:
        self.sound = np.zeros(int(self.samplerate * self.duration))