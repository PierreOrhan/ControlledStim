import voxpopuli
import numpy as np
from dataclasses import dataclass,field
from sounds.perExperiment.sound_elements import Sound
from scipy.io.wavfile import read, write
from io import BytesIO
import torch
from julius import  resample_frac

@dataclass
class FrenchSyllable(Sound):
    """ Generate syllable with a French voice at a given speed using MRBOLA+ESPEAK synthesizer from the voxpopuli package."""
    syllable : str = field(default="tu")
    speed : int = 160
    lang : str = "fr"
    voice_id : int = 1
    def __post_init__(self):
        self.name = "FrenchSyllable_"+self.syllable
        voice = voxpopuli.Voice(speed=160, lang="fr",voice_id=1)  # "es",voice_id=2
        pList = voice.to_phonemes(self.syllable)
        ## make sure there is no silence:
        to_keep = np.array([not p.name == "_" for p in pList],dtype=bool)
        old_duration = np.array([p.duration for p in pList])
        new_duration = old_duration / np.sum(old_duration[to_keep]) * self.duration

        newPlist = []
        for p, n, tokeep in zip(pList, new_duration, to_keep):
            if tokeep:
                p2 = p
                p2.duration = n*1000 # the duration of phoneme is in milliseconds.
                ## change to no pitch modifier:
                p2.pitch_modifiers = []
                newPlist += [p2]
        newPlist = voxpopuli.PhonemeList(newPlist)
        wav = voice.to_audio(newPlist)
        rate, wave_array = read(BytesIO(wav))
        wave_array = resample_frac(torch.tensor(np.array(wave_array, dtype="float")), rate, self.samplerate).detach().numpy()
        wave_array = wave_array / np.sqrt(np.sum(wave_array ** 2))
        self.sound = np.concatenate(
                    [wave_array, np.zeros(int(self.duration * 16000) - wave_array.shape[0], dtype=wave_array.dtype)])

@dataclass
class EnglishSyllable(Sound):
    """ Generate syllable with a French voice at a given speed using MRBOLA+ESPEAK synthesizer from the voxpopuli package."""
    syllable : str = field(default="tu")
    speed : int = 160
    lang : str = "en"
    voice_id : int = 1
    def __post_init__(self):
        self.name = "EnglishSyllable_"+self.syllable
        voice = voxpopuli.Voice(speed=self.speed, lang=self.lang,voice_id=self.voice_id)
        pList = voice.to_phonemes(self.syllable)
        ## make sure there is no silence:
        to_keep = np.array([not p.name == "_" for p in pList],dtype=bool)
        old_duration = np.array([p.duration for p in pList])
        new_duration = old_duration / np.sum(old_duration[to_keep]) * self.duration

        newPlist = []
        for p, n, tokeep in zip(pList, new_duration, to_keep):
            if tokeep:
                p2 = p
                p2.duration = n*1000 # the duration of phoneme is in milliseconds.
                ## change to no pitch modifier:
                p2.pitch_modifiers = []
                newPlist += [p2]
        newPlist = voxpopuli.PhonemeList(newPlist)
        wav = voice.to_audio(newPlist)
        rate, wave_array = read(BytesIO(wav))
        wave_array = resample_frac(torch.tensor(np.array(wave_array, dtype="float")), rate, self.samplerate).detach().numpy()
        wave_array = wave_array / np.sqrt(np.sum(wave_array ** 2))
        self.sound = np.concatenate(
                    [wave_array, np.zeros(int(self.duration * 16000) - wave_array.shape[0], dtype=wave_array.dtype)])