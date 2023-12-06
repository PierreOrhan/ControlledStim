import os.path
import shutil
from typing import Union
import pathlib
import numpy as np
import librosa
import zarr as zr
from sounds.api import SoundGenerator
import soundfile as sf
import random
import pandas as pd

RFRAM_key = None
Bip_Rand_key = None
test_dir = str(pathlib.Path(__file__).parent.parent.absolute() / "tests")
data_dir = str(pathlib.Path(__file__).parent.parent.absolute() / "data")


samplerates_sound = {
    "gaussian_N": 16000,
    "gaussian_RN": 16000,
    "gaussian_RefRN": 16000,
    "bip_": 16000,
    "silence": 16000,
    "Bip_Rand": 16000,
    "Bip_Ref_": 16000
}

durations_sound = {
    "gaussian_N": 1000,
    "gaussian_RN": 1000,
    "gaussian_RefRN": 1000,
    "bip_": 1000,
    "silence": 1000,
    "Bip_Rand": 50,
    "Bip_Ref_": 50
}
consines_rmp_length = {
    "gaussian_N": 5,
    "gaussian_RN": 5,
    "gaussian_RefRN": 5,
    "bip_": 5,
    "silence": 5,
    "Bip_Rand": 5,
    "Bip_Ref_": 5
}

from datasets import load_dataset, Audio, Dataset


def rand(nb_elements: int) -> list[str]:
    sequence = nb_elements*["Bip_Rand"]
    return sequence

def reg(nb_cyc: int, Rcyc:int) -> list[str]:
    sequence = []
    for i in range(Rcyc):
        sequence.append("Bip_Ref_"+str(i+1))
    sequence *= nb_cyc
    return sequence

def rfram_sequence_exp1() -> list[str]:
    """Generates a correct RFRAM_1 sequence.

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
                if 0 < j < len(sequence) - 1:
                    if sequence[j] != refRN and sequence[j + 1] != refRN and sequence[j - 1] != refRN:
                        sequence[i], sequence[j] = sequence[j], sequence[i]
                elif j == 0:
                    if sequence[j] != refRN and sequence[j + 1] != refRN:
                        sequence[i], sequence[j] = sequence[j], sequence[i]
                else:
                    if sequence[j] != refRN and sequence[j - 1] != refRN:
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
    "LocalGlobal_Deviant_1": ["bip_1", "bip_1", "bip_1", "bip_2"],
    "LocalGlobal_Deviant_2": ["bip_1", "bip_1", "bip_1", "bip_3"],
    "LocalGlobal_Omission": ["bip_1", "bip_1", "bip_1", "silence"],

    "RandReg_5": [],
    "RandReg_8": [],
    "RandReg_10": [],
    "RandReg_20": []
}

protocol_name_to_sequences = {
    "LocalGlobal_ssss": ["LocalGlobal_Standard", "LocalGlobal_Standard", "LocalGlobal_Standard", "LocalGlobal_Deviant_1"],
    "LocalGlobal_sssd": ["LocalGlobal_Deviant_1", "LocalGlobal_Deviant_1", "LocalGlobal_Deviant_1", "LocalGlobal_Deviant_2"],
    "LocalGlobal_sss_": ["LocalGlobal_Deviant_1", "LocalGlobal_Deviant_1", "LocalGlobal_Deviant_1", "LocalGlobal_Omission"],
    "RandomRegular": [],
    "SyllableStream": [],
    "Habituation": [],
    "TestRandom": [],
    "TestDeviant": []
}


class Sequence:
    name: str
    sounds: list[str]
    isi: float  # in seconds

    def __init__(self, name: str, isi: float):
        self.name = name
        self.isi = isi
        if name == "RFRAM_1":
            self.sounds = rfram_sequence_exp1()
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
        self.duration = duration  # ms

        if sound is None:
            if name == "gaussian_N":

                n_samples = int(samplerate * duration / 1000)
                noise = np.random.normal(0, 1, n_samples)

                # # Applying hamming window: necessary??
                # # creating ramps
                # hanning_window = np.hanning(consine_rmp_length / 1000 * samplerate)
                # hanning_window = hanning_window[:int(np.floor(hanning_window.shape[0] / 2))]
                # # filtering tones with ramps:
                # noise[:hanning_window.shape[0]] = noise[:hanning_window.shape[0]] * hanning_window
                # noise[-hanning_window.shape[0]:] = noise[-hanning_window.shape[0]:] * hanning_window[::-1]

                self.sound = noise

            elif name == "gaussian_RN" or name == "gaussian_RefRN":
                n_samples = int(samplerate * duration / 2000)
                noise = np.random.normal(0, 1, n_samples)
                r_noise = np.concatenate((noise, noise))

                # # Applying hamming window: necessary??
                # # creating ramps
                # hanning_window = np.hanning(consine_rmp_length / 1000 * samplerate)
                # hanning_window = hanning_window[:int(np.floor(hanning_window.shape[0] / 2))]
                # # filtering tones with ramps:
                # r_noise[:hanning_window.shape[0]] = r_noise[:hanning_window.shape[0]] * hanning_window
                # r_noise[-hanning_window.shape[0]:] = r_noise[-hanning_window.shape[0]:] * hanning_window[::-1]

                self.sound = r_noise

            elif name[0:4] == "bip_":
                # creating the tone with random distribution of frequency
                dis = np.random.uniform(0, 1, fs.shape[0])

                tone = np.transpose(np.transpose(
                    np.stack([librosa.tone(f, sr=samplerate, duration=duration / 1000) for f in fs],
                             axis=0)) @ np.transpose(dis))
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

            elif name[0:8] == "Bip_Ref_" or name == "Bip_Rand":
                # creating the tone with random distribution of frequency
                dis = np.random.uniform(0, 1, fs.shape[0])

                tone = np.transpose(np.transpose(
                    np.stack([librosa.tone(f, sr=samplerate, duration=duration / 1000) for f in fs],
                             axis=0)) @ np.transpose(dis))
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

            elif name == "silence":

                silence = np.zeros(int(samplerate * duration/1000))

                self.sound = silence

            else:
                print("name: ", name)
                raise ValueError("'name' is not defined")


class Sound_pool:
    name: str
    sounds: dict[str, Sound]
    fs: np.ndarray

    def __init__(self, name, fs=np.logspace(np.log(222), np.log(2000), 20, base=np.exp(1)), sounds=None):
        self.name = name
        self.sounds = sounds
        self.fs = fs

        if sounds == None:
            self.sounds = {}


    def __add__(self, other):
        if isinstance(other, Sound_pool):
            return Sound_pool(self.name + other.name, self.fs, {**self.sounds, **other.sounds})
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(self.__class__, other.__class__))


    def add(self, sound : Sound):
        self.sounds[sound.name] = sound

    def addSequence(self, seq: Sequence):
        for key in seq.sounds:
            if key in self.sounds.keys():
                continue
            # gaussian_N and gaussian_RN have to be different each time
            elif key == "gaussian_N" or key == "gaussian_RN":
                self.sounds[key] = RFRAM_key
            elif key == "Bip_Rand":
                self.sounds[key] = Bip_Rand_key
            elif key[0:4] == "bip_":
                self.sounds[key] = Sound(key, samplerates_sound[key[0:4]], durations_sound[key[0:4]],
                                         consines_rmp_length[key[0:4]], self.fs)
            elif key[0:8] == "Bip_Ref_":
                self.sounds[key] = Sound(key, samplerates_sound[key[0:8]], durations_sound[key[0:8]],
                                         consines_rmp_length[key[0:8]], self.fs)
            else:
                self.sounds[key] = Sound(key, samplerates_sound[key], durations_sound[key],
                                         consines_rmp_length[key], self.fs)


class Combinator:
    name: str
    dirWav: str
    samplerate: int


    def __init__(self, name: str, samplerate: int, dirWav: str = data_dir):
        self.name = name
        self.dirWav = str(pathlib.Path(dirWav) / name)
        self.samplerate = samplerate

    def combine(self, seq_list: list[Sequence], sound_pool: Sound_pool):
        """Combine the sounds in the sequence and save the resulting sound in the directory dirWav."""
        name = []
        wav_path = []
        mask_info_path = []
        duration = []
        isi = []
        sound_info_path = []

        shutil.rmtree(self.dirWav, ignore_errors=True)
        os.makedirs(self.dirWav, exist_ok=True)

        for seq in seq_list:
            sound_seq = []
            for sound in seq.sounds:
                # gaussian_N and gaussian_RN have to be different each time
                if sound_pool.sounds[sound] == RFRAM_key or sound_pool.sounds[sound] == Bip_Rand_key:
                    newSound = Sound(sound, samplerates_sound[sound], durations_sound[sound],
                                     consines_rmp_length[sound], sound_pool.fs)
                    sound_seq.append(newSound.sound)
                else:
                    sound_seq.append(sound_pool.sounds[sound].sound)
                sound_seq.append(np.zeros(int(self.samplerate * seq.isi)))
                # concatenate the sounds
            sound_seq = np.concatenate(sound_seq)

            # save the sound
            os.makedirs(self.dirWav, exist_ok=True)
            sf.write(os.path.join(self.dirWav, seq.name + ".wav"), sound_seq, samplerate=self.samplerate)

            # save the sound info
            name.append(seq.name)
            wav_path.append(str(os.path.join(self.dirWav, seq.name + ".wav")))
            mask_info_path.append(str(os.path.join(self.dirWav, seq.name + ".masks")))
            duration.append(len(sound_seq)/self.samplerate)
            isi.append(seq.isi)
            sound_info_path.append(os.path.join(self.dirWav, "sound_info", seq.name + ".csv"))

            # generating a csv with the specific sounds info concerning the sequence
            start = []
            for i in range(len(seq.sounds)):
                if i == 0:
                    start.append(0)
                else:
                    if seq.sounds[i-1][0:4] == "bip_":
                        ds = durations_sound["bip_"]/1000
                    elif seq.sounds[i-1][0:8] == "Bip_Ref_":
                        ds = durations_sound["Bip_Ref_"]/1000
                    else:
                        ds = durations_sound[seq.sounds[i-1]]/1000
                    start.append(start[i-1] + ds + seq.isi)
            df_sound_info = pd.DataFrame()
            df_sound_info["name"] = seq.sounds
            df_sound_info["start"] = start
            durations = []
            for sound in seq.sounds:
                if sound[0:4] == "bip_":
                    ds = durations_sound["bip_"] / 1000
                elif sound[0:8] == "Bip_Ref_":
                    ds = durations_sound["Bip_Ref_"] / 1000
                else:
                    ds = durations_sound[sound] / 1000
                durations.append(ds)
            df_sound_info["duration"] = durations

            os.makedirs(os.path.join(self.dirWav, "sound_info"), exist_ok=True)
            df_sound_info.to_csv(os.path.join(self.dirWav, "sound_info", seq.name + ".csv"), index=False)

        # generating a csv with the sequence names and the corresponding soundfile paths
        df = pd.DataFrame()
        df["name"] = name
        df["wav_path"] = wav_path
        df["mask_info_path"] = mask_info_path
        df["duration"] = duration
        df["isi"] = isi
        df["sound_info_path"] = sound_info_path

        df.to_csv(os.path.join(self.dirWav, "sequences.csv"), index=False)


class Protocol:
    name: str
    sequence_isi: float

    def __init__(self, name: str, sequence_isi: float):
        self.name = name
        self.sequence_isi = sequence_isi

    def create(self, soundseq_dataset_csv: str):
        df = pd.read_csv(soundseq_dataset_csv, index_col=False)
        ann_dict = {
            "audio_data": [],
            "sample_rate": [],
            "label": []
        }
        for seq in protocol_name_to_sequences[self.name]:
            data, sr = sf.read(df[df["name"] == seq]["wav_path"].values[0])
            ann_dict["audio_data"].append(data)
            ann_dict["sample_rate"].append(sr)
            ann_dict["label"].append(seq)

        ann_dataset = Dataset.from_dict(ann_dict)
        return ann_dataset
