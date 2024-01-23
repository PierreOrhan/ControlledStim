import voxpopuli
import numpy as np
from dataclasses import dataclass,field
from sounds.perExperiment.sound_elements import Sound
from scipy.io.wavfile import read, write
from io import BytesIO
import torch
from julius import  resample_frac
from typing import Tuple

@dataclass
class FrenchSyllable(Sound):
    """ Generate syllable with a French voice at a given speed using MRBOLA+ESPEAK synthesizer from the voxpopuli package."""
    syllable : str = field(default="tu")
    speed : int = 160
    lang : str = "fr"
    voice_id : int = 1
    pitch_modifiers : list[Tuple[float,float]] = field(default_factory=list)
    force_duration : bool = True
    def __post_init__(self):
        self.name = self.__class__.__name__+"_"+self.syllable
        voice = voxpopuli.Voice(speed=160, lang=self.lang,voice_id=self.voice_id)  # "es",voice_id=2
        pList = voice.to_phonemes(self.syllable)
        ## make sure there is no silence:
        to_keep = np.array([not p.name == "_" for p in pList],dtype=bool)
        old_duration = np.array([p.duration for p in pList])
        new_duration = old_duration / np.sum(old_duration[to_keep]) * self.duration

        newPlist = []
        for p, n, tokeep in zip(pList, new_duration, to_keep):
            if tokeep:
                p2 = p
                if self.force_duration:
                    p2.duration = n*1000 # the duration of phoneme is in milliseconds.
                ## change to no pitch modifier:
                p2.pitch_modifiers = self.pitch_modifiers
                newPlist += [p2]
        newPlist = voxpopuli.PhonemeList(newPlist)
        wav = voice.to_audio(newPlist)
        rate, wave_array = read(BytesIO(wav))
        wave_array = resample_frac(torch.tensor(np.array(wave_array, dtype="float")), rate, self.samplerate).detach().numpy()
        wave_array = wave_array / np.sqrt(np.mean(wave_array ** 2))
        if self.force_duration:
            self.sound = np.concatenate(
                        [wave_array, np.zeros(int(self.duration * 16000) - wave_array.shape[0], dtype=wave_array.dtype)])
        else:
            self.sound = wave_array
            self.duration = self.sound.shape[0]/self.samplerate

@dataclass
class EnglishSyllable(FrenchSyllable):
    lang : str = "en"