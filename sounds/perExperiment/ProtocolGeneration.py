import os.path
from typing import Union
import pathlib
import numpy as np
import librosa
import zarr as zr
from sounds.api import SoundGenerator
import soundfile as sf
import random


samplerates_sound = {
    "gaussian_N": 16000,
    "gaussian_RN": 16000,
    "gaussian_RefN": 16000,
    "bip_1": 16000,
    "bip_2": 16000,
}

list_sounds_name = list(samplerates_sound)

durations_sound = {
    "gaussian_N": 1000,
    "gaussian_RN": 1000,
    "gaussian_RefN": 1000,
    "bip_1": 1000,
    "bip_2": 1000,
}
consines_rmp_length = {
    "gaussian_N": 5,
    "gaussian_RN": 5,
    "gaussian_RefN": 5,
    "bip_1": 5,
    "bip_2": 5,
}

from datasets import load_dataset, Audio

def rfram_sequence() -> list[str]:
    """Generates a correct RFRAM sequence.

    :return: list[str]
    """

    nbN = 50
    nbRN = 100
    nbRefRN = 50
    N = "gaussian_N"
    RN = "gaussian_RN"
    refRN = "gaussian_RefRN"

    sequence = [N] * nbN + [RN] * nbRN + [refRN] * nbRefRN
    random.shuffle(sequence)

    for i in range(1, len(sequence)):
        if sequence[i] == refRN and sequence[i - 1] == refRN:
            while sequence[i] == refRN:
                j = random.randint(0, len(sequence) - 1)
                if sequence[j] != refRN:
                    sequence[i], sequence[j] = sequence[j], sequence[i]

    return sequence


sequence_name_to_sounds = {
    "LOT_repeat": [],
    "LOT_alternate": [],
    "LOT_pairs": [],
    "LOT_quadruplets": [],
    "LOT_PairsAndAlt1": [],
    "LOT_Shrinking": [],
    "LOT_PairsAndAlt2": [],
    "LOT_threeTwo": [],
    "LOT_centermirror": [],
    "LOT_complex": [],
    # "tree1":tree1,
    # "tree2":tree2,

    "LocalGlobal_Standard": ["bip_1", "bip_1", "bip_1", "bip_1"],
    "LocalGlobal_Deviant": ["bip_1", "bip_1", "bip_1", "bip_2"],
    "LocalGlobal_Omission": ["bip_1", "bip_1", "bip_1", "silence"],

    "RandReg_5": [],
    "RandReg_8": [],
    "RandReg_10": [],
    "RandReg_20": [],

}

class Sequence:
    name: str
    sounds: list[str]
    isi: float  # in seconds

    def __init__(self, name: str, isi: float):
        self.name = name
        self.isi = isi
        if name == "RFRAM":
            self.sounds = rfram_sequence()
        else:
            self.sounds = sequence_name_to_sounds[name]


class Sound:
    name: str
    sound: np.ndarray
    samplerate: int
    duration: int

    def __init__(self, name, samplerate, duration, consine_rmp_length, fs=None, sound=None):
        self.name = name
        self.sound = sound
        self.samplerate = samplerate
        self.duration = duration #ms

        if sound is None:
            if name == "gaussian_N":
                n_samples = samplerate * duration
                noise = np.random.normal(0, 1, n_samples)

                # Applying hamming window: necessary??
                # creating ramps
                hanning_window = np.hanning(consine_rmp_length / 1000 * samplerate)
                hanning_window = hanning_window[:int(np.floor(hanning_window.shape[0] / 2))]
                # filtering tones with ramps:
                noise[:hanning_window.shape[0]] = noise[:hanning_window.shape[0]] * hanning_window
                noise[-hanning_window.shape[0]:] = noise[-hanning_window.shape[0]:] * hanning_window[::-1]

                self.sound = noise

            elif name[:10] == "gaussian_R":
                n_samples = int(samplerate * duration / 2)
                noise = np.random.normal(0, 1, n_samples)
                r_noise = np.concatenate((noise, noise))

                # Applying hamming window: necessary??
                # creating ramps
                hanning_window = np.hanning(consine_rmp_length / 1000 * samplerate)
                hanning_window = hanning_window[:int(np.floor(hanning_window.shape[0] / 2))]
                # filtering tones with ramps:
                r_noise[:hanning_window.shape[0]] = r_noise[:hanning_window.shape[0]] * hanning_window
                r_noise[-hanning_window.shape[0]:] = r_noise[-hanning_window.shape[0]:] * hanning_window[::-1]

                self.sound = r_noise

            elif name[:3] == "bip":
                tone = np.sum(np.stack([librosa.tone(f, sr=samplerate, duration=duration / 1000) for f in fs], axis=0),axis=0)
                # tone = librosa.tone(f, sr=samplerate, duration=duration_tone/1000)

                # creating ramps
                hanning_window = np.hanning(consine_rmp_length / 1000 * samplerate)
                hanning_window = hanning_window[:int(np.floor(hanning_window.shape[0] / 2))]
                # filtering tones with ramps:
                tone[:hanning_window.shape[0]] = tone[:hanning_window.shape[0]] * hanning_window
                tone[-hanning_window.shape[0]:] = tone[-hanning_window.shape[0]:] * hanning_window[::-1]

                # we normalize, to see if this could help the network to perform better!
                ## Normalization:
                tone = tone / np.sqrt(np.sum(tone ** 2, axis=-1, keepdims=True))

                self.sound = tone

            else:
                raise ValueError("'name' is not defined")




class Sound_pool:
    name: str
    sounds: dict[str, Sound]

    def __init__(self, name, fs=np.logspace(np.log(222), np.log(2000), 20, base=np.exp(1)), sounds=None):
        self.name = name
        self.sounds = sounds

        if sounds is None:
            dict_sounds = {}
            for key in list_sounds_name:
                dict_sounds[key] = Sound(key, samplerates_sound[key], durations_sound[key], consines_rmp_length[key], fs)

            self.sounds = dict_sounds




class Combinator:
    name: str

    def __init__(self, name: str):
        self.name = name

    def combine(self, seq_list: list[Sequence], sound_pool: Sound_pool):
        for seq in seq_list:
            pass

class Protocol:
    name: str
    sequence_isi: float
