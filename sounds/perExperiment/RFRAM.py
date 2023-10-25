import os.path
from typing import Union
import pathlib
import numpy as np
import librosa
import zarr as zr
from sounds.api import SoundGenerator
import soundfile as sf

from sounds.experimentsClass.elementMasking import ElementMasking

## Defines different structure manipulation routines:
# def duplicate(X : np.ndarray, n :int  ) -> np.ndarray:
#     # duplicate X for n times, producing (n+1)X
#     return np.concatenate([X for _  in range(n+1)])
# def inner_repeat(X : np.ndarray, n : int ) -> np.ndarray:
#     return np.repeat(X,n)
# def merge(X: np.ndarray,Y: np.ndarray) -> np.ndarray:
#     return np.concatenate([X,Y])
# def roll(X:np.ndarray,n : int = 4) -> np.ndarray:
#     return np.roll(X,shift=n)
# def mirror(X: np.ndarray) -> np.ndarray :
#     return np.concatenate([X,X[::-1]])

from typing import List
def duplicateL(X : List[np.ndarray], n :int  ) -> List[np.ndarray]:
    # duplicate X for n times, producing (n+1)X
    out = []
    for _ in range(n+1):
        out += X
    return out
def mergeL(X: List[np.ndarray],Y: List[np.ndarray]) -> List[np.ndarray]:
    return X+Y

def Exp1(X, Y):
    pass


### Defines all the sequences

# RFRAM paper sequence

from dataclasses import dataclass
from typing import Dict
@dataclass
class SeqsInfo:
    seqs : Dict = None
    soa : Dict = None # silence between stimulis
    nbN : Dict = None
    nbRN : Dict = None
    nbRefRN : Dict = None

def get_all_sequences():
    seqs = {
        "Exp1":Exp1}

    # To retrieve the original LOT experimentsClass with 16 tones, sequences are composed of 16 tones,
    # which are often composed of a repetition of a sub-motif
    # we indicate here the amount of repetition of that motif:

    soa = {
           "Exp1" : 1000}

    nbN = {
        "Exp1": 50}

    nbRN = {
        "Exp1": 100}

    nbRefRN = {
        "Exp1": 50}

    return SeqsInfo(seqs, soa, nbN, nbRN, nbRefRN)

class RFRAM_structure(SoundGenerator,ElementMasking):

    @classmethod
    def get_info(cls, with_dict=False):

        samplerate = 16000  # 44000 hz in the original study, here using 16000 for wav2vec2

        # ## 23/08/2023: We try to see if having more complicated sounds
        # # (here some with multiples frequencies, could change what is happening:
        # frequencies_fundamental = np.logspace(np.log(222),np.log(500),20,base = np.exp(1))
        # frequencies = np.array([[f,f*2,f*4] for f in frequencies_fundamental])

        cyc_names = ["block1"]

        # nb_Rcyc = 8 # NOTE: in the original experiment this is varied between 7 or 8

        n_gaussian = 10
        range_mean = [-0.5, 0.5]
        range_sd = [0, 0.5]

        if with_dict:
            return {"samplerate": samplerate, "cyc_names": cyc_names,
                    "n_gaussian": n_gaussian, "range_mean": range_mean, "range_sd": range_sd}

        return samplerate, cyc_names, n_gaussian, range_mean, range_sd

    @classmethod
    def _N(cls, samplerate, n_gaussian, range_mean, range_sd):

        seqInfo = cls.get_all_seq()

        mean = np.random.uniform(range_mean[0], range_mean[1])
        stddev = np.random.uniform(range_sd[0], range_sd[1])

        noise = np.random.normal(mean, stddev, (n_gaussian, samplerate))
        noise = np.sum(noise, axis=0)

        blank = np.zeros((1, len(noise)))
        repeated_noise = np.concatenate((noise, blank))

        return noise

    @classmethod
    def _RN(cls, samplerate, n_gaussian, range_mean, range_sd):

        seqInfo = cls.get_all_seq()

        mean = np.random.uniform(range_mean[0], range_mean[1])
        stddev = np.random.uniform(range_sd[0], range_sd[1])

        noise = np.random.normal(mean, stddev, (n_gaussian, samplerate/2))
        noise = np.sum(noise, axis=0)
        repeated_noise = np.concatenate((noise, noise))

        blank = np.zeros((1, len(repeated_noise)))
        repeated_noise = np.concatenate((repeated_noise, blank))

        return repeated_noise

    @classmethod
    def get_all_seq(cls):
        return get_all_sequences()

    @classmethod
    def generateSounds(cls,dirWav: Union[str,pathlib.Path], dirZarr: Union[str,pathlib.Path]):

        samplerate, cyc_names, n_gaussian, range_mean, range_sd = cls.get_info()

        seqInfo = cls.get_all_seq()

        exception = {}

        # Block One:
        for blockName in cyc_names:
            list_seq = list(seqInfo.seqs.keys())
            if blockName in exception.keys():
                list_seq = np.setdiff1d(list_seq,exception[blockName])
            for seqid in list_seq:

                fileName = blockName+"_"+seqid

                sound_mat = np.array([])
                sound_block = np.array([])
                ref_noise = cls._RN(samplerate, n_gaussian, range_mean, range_sd)
                last = -1
                for i in range(seqInfo.nbN[seqid]+seqInfo.nbRN[seqid]+seqInfo.nbRefRN[seqid]):
                    current = np.random.randint(0, 4)
                    if last == 3:
                        while current == 3:
                            current = np.random.randint(0, 4)
                    if current == 1:
                        sound_mat = np.concatenate((sound_mat, cls._N(samplerate, n_gaussian, range_mean, range_sd)))
                    elif current == 2:
                        sound_mat = np.concatenate((sound_mat, cls._RN(samplerate, n_gaussian, range_mean, range_sd)))
                    else:
                        sound_mat = np.concatenate((sound_mat, ref_noise))

                    sound_block = np.concatenate((sound_block, current))
                    last = current

                sound_names = ["exp_1"]
                sound_names = np.array(sound_names)

                os.makedirs(os.path.join(dirZarr, fileName),exist_ok=True)
                zg = zr.open_group(os.path.join(dirZarr, fileName, "sounds.zarr"), mode="w")
                zg.array("sound_mat", data=sound_mat, chunks=(1, None))
                zg.array("sound_names", data=np.array([k + ".wav" for k in sound_names]), chunks=(None,))
                zg.array("tones_sequences", data=sound_block, chunks=(None,))

                os.makedirs(os.path.join(dirWav, fileName),exist_ok=True)
                for sname,s in zip(sound_names,sound_mat):
                    sf.write(os.path.join(dirWav,fileName,sname+".wav"),s,samplerate=samplerate)


    @classmethod
    def generateWav2vec2Mask(cls,dirZarr,oneEvalPerEvent=True):

        samplerate, cyc_names, n_gaussian, range_mean, range_sd = cls.get_info()
        seqInfo = cls.get_all_seq()

        exception = {}
        for blockName in cyc_names:
            list_seq = list(seqInfo.seqs.keys())
            if blockName in exception.keys():
                list_seq = np.setdiff1d(list_seq, exception[blockName])
            for seqid in list_seq:
                duration_silence = seqInfo.soa[seqid]
                for vary_inner_sequence,typeTask in zip([True,False],["Detect","Generalize"]):
                    fileName = blockName+"_"+seqid+"_"+typeTask
                    cls._mask_and_latent(dirZarr,fileName,duration_tone,oneEvalPerEvent,
                                         duration_silence=duration_silence)
