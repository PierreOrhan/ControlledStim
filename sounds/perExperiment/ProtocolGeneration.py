import os.path
from typing import Union
import pathlib
import numpy as np
import librosa
import zarr as zr
from sounds.api import SoundGenerator
import soundfile as sf

from datasets import load_dataset, Audio

def rfram_sequence() -> list[str]:
    """Generates a correct RFRAM sequence.

    :return: list[str]
    """
    sequence = []
    last = -1
    nb_each = {
        "N": 0,
        "RN": 0,
        "refRN": 0
    }
    nbN = 50
    nbRN = 100
    nbRefRN = 50
    for i in range(nbN + nbRN + nbRefRN):
        current = np.random.randint(1, 4)
        if last == 3:
            while current == 3:
                current = np.random.randint(0, 4)
        if current == 1 and nb_each["N"] < nbN:
            sequence.append("gaussian_N")
            nb_each["N"] += 1
        elif current == 2 and nb_each["RN"] < nbRN:
            sequence.append("gaussian_RN")
            nb_each["RN"] += 1
        elif nb_each["refRN"] < nbRefRN:
            sequence.append("gaussian_RefRN")
            nb_each["refRN"] += 1
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


class Sound_pool:
    name: str
    sounds: dict[Sound]


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
