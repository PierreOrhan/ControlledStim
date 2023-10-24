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
    seqs_number_element : Dict = None
    seqs_subsize_add : Dict = None
    seqs_inner_repeat: Dict =None
    seqs_deviants_pos: Dict = None
    soa : Dict = None # silence between stimulis
    seqs_diversity : Dict[str,int] = None # constraint the size of the original alphabet when sampling inside a sequence

def get_all_sequences():
    seqs = {
        "Exp1":Exp1}
    seqs_number_element = {
            "Exp1":2}

    seqs_subsize_add = {
        "Exp1":[0,0]}
            # we can vary the size of the inner motives by adding to the initial size...

    # To retrieve the original LOT experimentsClass with 16 tones, sequences are composed of 16 tones,
    # which are often composed of a repetition of a sub-motif
    # we indicate here the amount of repetition of that motif:
    seqs_inner_repeat = {
            "Exp1":8}
    seqs_deviants_pos = {
               "Exp1" : [0]}

    soa = {
           "Exp1" : 1000}

    return SeqsInfo(seqs,seqs_number_element,seqs_subsize_add,seqs_inner_repeat,seqs_deviants_pos,soa)

class RFRAM_structure(SoundGenerator,ElementMasking):

    @classmethod
    def get_info(cls, with_dict=False):

        samplerate = 16000  # 44000 hz in the original study, here using 16000 for wav2vec2

        # ## 23/08/2023: We try to see if having more complicated sounds
        # # (here some with multiples frequencies, could change what is happening:
        # frequencies_fundamental = np.logspace(np.log(222),np.log(500),20,base = np.exp(1))
        # frequencies = np.array([[f,f*2,f*4] for f in frequencies_fundamental])

        duration_tone = 500  # ms, duration of the tone
        consine_rmp_length = 5  # ms, duration of the upper and lower cosine ramp

        # Rcyc = [1,5,10] #15,20,40
        # cyc_names = ["block1","block5","block10"]
        Rcyc = [1]
        cyc_names = ["block1"]

        # nb_Rcyc = 8 # NOTE: in the original experiment this is varied between 7 or 8

        if with_dict:
            return {"Rcyc": Rcyc, "samplerate": samplerate,
                    "duration_tone": duration_tone, "consine_rmp_length": consine_rmp_length}

        return Rcyc, samplerate, duration_tone, consine_rmp_length, cyc_names

    @classmethod
    def _N(cls, samplerate, n_gaussian, range_mean, range_sd):

        mean = np.random.uniform(range_mean[0], range_mean[1])
        stddev = np.random.uniform(range_sd[0], range_sd[1])

        noise = np.random.normal(mean, stddev, (n_gaussian, samplerate))
        noise = np.sum(noise, axis=0)

        return noise

    @classmethod
    def _RN(cls, samplerate, n_gaussian, range_mean, range_sd):

        mean = np.random.uniform(range_mean[0], range_mean[1])
        stddev = np.random.uniform(range_sd[0], range_sd[1])

        noise = np.random.normal(mean, stddev, (n_gaussian, samplerate/2))
        noise = np.sum(noise, axis=0)
        repeated_noise = np.concatenate((noise, noise))

        return repeated_noise

    @classmethod
    def get_all_seq(cls):
        return get_all_sequences()

    @classmethod
    def generateSounds(cls,dirWav: Union[str,pathlib.Path], dirZarr: Union[str,pathlib.Path]):

        Rcyc,samplerate,duration_tone,consine_rmp_length,cyc_names = cls.get_info()

        seqInfo = cls.get_all_seq()

        exception = {}

        # Block One:
        for blockName,rcyc in zip(cyc_names,Rcyc):
            list_seq = list(seqInfo.seqs.keys())
            if blockName in exception.keys():
                list_seq = np.setdiff1d(list_seq,exception[blockName])
            for seqid in list_seq:

                fileName = blockName+"_"+seqid

                # FIRST the set of sounds with cycle of size Rcyc = 10
                # We decide to work with sounds of constant size in the network case: 8
                # nbRandreg = 20 #50
                nbRegrand = 10 #50
                nbRegDeviant = 5

                # sound_mat = [cls._RANDREG(freq,rcyc,seqInfo,seqid,vary_outer_sequence=vary_outer_sequence) for _ in range(nbRandreg)]
                # sound_names = ["randreg_"+str(i) for i in range(nbRandreg)]
                sound_mat = [cls._REGRAND(freq,rcyc,seqInfo,seqid,vary_outer_sequence=vary_outer_sequence,
                                          number_out_motif =3) for _ in range(nbRegrand)]
                sound_names = ["regrand_"+str(i) for i in range(nbRegrand)]
                for i in range(nbRegDeviant):
                    sound_mat += cls._REGDEVIANT(freq,rcyc,seqInfo,seqid,vary_outer_sequence=vary_outer_sequence,
                                                 number_out_motif =3)
                    sound_names += ["regdeviant_"+str(i)+"_"+str(switchid) for switchid in range(len(seqInfo.seqs_deviants_pos[seqid]))]

                # sound_mat += [cls._CST(freq,rcyc,nb_Rcyc) for _ in range(15)]
                # sound_names += ["cst_"+str(i) for i in range(15)]

                sound_block = np.array([s[0] for s in sound_mat],dtype=int)
                sound_names = np.array(sound_names)

                sound_mat = np.stack([tones[sound_block[i,:],:].reshape(-1) for i in range(sound_block.shape[0])],axis=0)

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

        Rcyc,samplerate,duration_tone,consine_rmp_length,cyc_names = cls.get_info()
        seqInfo = cls.get_all_seq()

        exception = {"block1":["LOT_repeat"]}  #{}
        for blockName,rcyc in zip(cyc_names,Rcyc):
            list_seq = list(seqInfo.seqs.keys())
            if blockName in exception.keys():
                list_seq = np.setdiff1d(list_seq, exception[blockName])
            for seqid in list_seq:
                duration_silence = seqInfo.soa[seqid]
                for vary_inner_sequence,typeTask in zip([True,False],["Detect","Generalize"]):
                    fileName = blockName+"_"+seqid+"_"+typeTask
                    cls._mask_and_latent(dirZarr,fileName,duration_tone,oneEvalPerEvent,
                                         duration_silence=duration_silence)
