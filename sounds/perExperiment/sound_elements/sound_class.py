import numpy as np
from dataclasses import dataclass,field

@dataclass(frozen=False)
class Sound:
    name: str = field(default_factory=str)
    samplerate: int = 16000
    duration: float = 0.05 # in seconds
    sound: np.ndarray = field(init=False)
    def __post_init__(self) -> None:
        self.sound = np.zeros(int(self.duration*self.samplerate))

def ramp_sound(s:Sound,cosine_rmp_length:float = 0.005) -> None:
    # Warning: modify in place the sounds.
    # creating ramps
    hanning_window = np.hanning(int(cosine_rmp_length * s.samplerate))
    hanning_window = hanning_window[:int(np.floor(hanning_window.shape[0] / 2))]
    # filtering tones with ramps:
    s.sound[:hanning_window.shape[0]] = s.sound[:hanning_window.shape[0]] * hanning_window
    s.sound[-hanning_window.shape[0]:] = s.sound[-hanning_window.shape[0]:] * hanning_window[::-1]

def normalize_sound(s:Sound) -> None:
    # warning: modify in place the sounds.
    s.sound = s.sound / np.sqrt(np.sum(s.sound ** 2, axis=-1, keepdims=True))


@dataclass(frozen=False)
class Sound_pool(list[Sound]):
    # The sound_pool is simply a list of Sound elements, but we define
    # specific methods to it.
    def __post_init__(self):
        self.picked = []

    @classmethod
    def from_list(cls,ls):
        s = Sound_pool()
        for l in ls:
            s.append(l)
        return s
    def pick_norepeat(self) -> Sound:
        ## memory cach the sound that were picked so that we can sample without repeat:
        pick = np.random.choice(np.setdiff1d(range(self.__len__()),self.picked),1)[0]
        self.picked.append(pick)
        return self[pick]
    def pick_norepeat_n(self,n:int) -> list[Sound]:
        ## memory cach the n sounds that were picked so that we can sample without repeat:
        picks = np.random.choice(np.setdiff1d(range(self.__len__()),self.picked),n,replace=False)
        for p in picks:
            self.picked.append(p)
        return [self[p] for p in picks]
    def clear_picked(self):
        self.picked = []
