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


### Defines all the sequences

# LOT paper sequence
def LOT_repeat(X,Y):
    return duplicateL([X],1)
def LOT_alternate(X,Y):
    return duplicateL(mergeL([X],[Y]),1)
def LOT_pairs(X,Y):
    return mergeL(duplicateL([X],1),duplicateL([Y],1))
def LOT_quadruplets(X,Y):
    return mergeL(duplicateL([X],3),duplicateL([Y],3))
def LOT_PairsAndAlt1(X,Y):
    return mergeL(mergeL(duplicateL([X],1),duplicateL([Y],1)),duplicateL(mergeL([X],[Y]),1))

def LOT_Shrinking(X,Y):
    out = []
    for i in np.arange(2, 4):
        out += mergeL(duplicateL([X],i),duplicateL([Y],i))
    return mergeL(out,duplicateL(mergeL([X],[Y]),1))
def LOT_complex(X,Y):
    return [X,Y,X,X,X,Y,Y,Y,Y,X,Y,Y,X,X,X,Y]
def LOT_centerMirror(X,Y):
    return [X,X,X,X,Y,Y,X,Y,X,Y,X,X,Y,Y,Y,Y]

def LOT_PairsAndAlt2(X,Y):
    return mergeL(mergeL([X],[Y]),mergeL(mergeL(duplicateL([X],1),duplicateL([Y],1)),mergeL([X],[Y])))
def LOT_threeTwo(X,Y):
    return mergeL(mergeL(mergeL(duplicateL([X],2),duplicateL([Y],1)),[X]),duplicateL([Y],1))

def LocalGlobal_Standard(X,Y):
    return duplicateL([X], 4)
def LocalGlobal_Deviant(X,Y):
    return mergeL(duplicateL([X], 3),[Y])

def LocalGlobal_Omission(X,Y):
    return mergeL(duplicateL([X], 3), [[-1]])

def RandReg_Rcyc(*args):
    # RandReg but Rcyc is to be defined.
    out = []
    for X in args:
        out += [X]
    return out

# A new tree sequence:
def tree1(X : np.ndarray, Y: np.ndarray, Z: np.ndarray):
    # xxxYxxxYZ --> minimal duplication should be:
    ## xxxYxxxYZxxxYxxxYZ (18)
    return mergeL(duplicateL(mergeL(duplicateL([X],2),[Y]),1),[Z])
def tree2(X : np.ndarray, Y: np.ndarray):
    ## xxxYxxxYxxxx (12) xxxYxxxYxxxx (18)
    # --> A very short local Global!
    return mergeL(duplicateL(mergeL(duplicateL([X],2),[Y]),1),duplicateL([X],3))


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
    seqs = {"LOT_repeat":LOT_repeat,
            "LOT_alternate":LOT_alternate,
            "LOT_pairs":LOT_pairs,
            "LOT_quadruplets":LOT_quadruplets,
            "LOT_PairsAndAlt1":LOT_PairsAndAlt1,
            "LOT_Shrinking" : LOT_Shrinking,
            "LOT_PairsAndAlt2":LOT_PairsAndAlt2,
            "LOT_threeTwo":LOT_threeTwo,
            "LOT_centermirror":LOT_centerMirror,
            "LOT_complex": LOT_complex,
            # "tree1":tree1,
            # "tree2":tree2,
            "LocalGlobal_Standard":LocalGlobal_Standard,
            "LocalGlobal_Deviant":LocalGlobal_Deviant,

            "LocalGlobal_Omission":LocalGlobal_Omission,

            "RandReg_5":RandReg_Rcyc,
            "RandReg_8": RandReg_Rcyc,
            "RandReg_10":RandReg_Rcyc,
            "RandReg_20":RandReg_Rcyc}
    seqs_number_element = {
            "LOT_repeat":2,
            "LOT_alternate":2,
            "LOT_pairs":2,
            "LOT_quadruplets":2,
            "LOT_PairsAndAlt1":2,
            "LOT_Shrinking": 2,
            "LOT_PairsAndAlt2":2,
            "LOT_threeTwo":2,
            "LOT_centermirror":2,
            "LOT_complex":2,
            # "tree1":3,
            # "tree2":2,
            "LocalGlobal_Standard":2,
            "LocalGlobal_Deviant":2,

            "LocalGlobal_Omission": 2,

            "RandReg_5":5,
            "RandReg_8":8,
            "RandReg_10":10,
            "RandReg_20":20}

    seqs_subsize_add = {"LOT_repeat":[0,0],
            "LOT_alternate":[0,0],
            "LOT_pairs":[0,0],
            "LOT_quadruplets":[0,0],
            "LOT_PairsAndAlt1":[0,0],
            "LOT_Shrinking": [0,0],
            "LOT_PairsAndAlt2":[0,0],
            "LOT_threeTwo":[0,0],
            "LOT_centermirror":[0,0],
            "LOT_complex":[0,0],
            # "tree1":[0,0,0],
            # "tree2":[0,0],
            "LocalGlobal_Standard": [0,0],
            "LocalGlobal_Deviant": [0,0],

            "LocalGlobal_Omission": [0, 0],

            "RandReg_5":[0 for _ in range(5)],
            "RandReg_8": [0 for _ in range(8)],
            "RandReg_10":[0 for _ in range(10)],
            "RandReg_20":[0 for _ in range(20)]}
            # we can vary the size of the inner motives by adding to the initial size...

    # To retrieve the original LOT experimentsClass with 16 tones, sequences are composed of 16 tones,
    # which are often composed of a repetition of a sub-motif
    # we indicate here the amount of repetition of that motif:
    seqs_inner_repeat = {
            "LOT_repeat":8,
            "LOT_alternate":4,
            "LOT_pairs":4,
            "LOT_quadruplets":2,
            "LOT_PairsAndAlt1":2,
            "LOT_Shrinking": 1,
            "LOT_PairsAndAlt2":2,
            "LOT_threeTwo":2,
            "LOT_centermirror":1,
            "LOT_complex":1,
            # "tree1":3,
            # "tree2":3,
            "LocalGlobal_Standard": 3,
            "LocalGlobal_Deviant": 3,

            "LocalGlobal_Omission": 3,

            "RandReg_5":4,
            "RandReg_8": 3,
            "RandReg_10":2,
            "RandReg_20":1}
    seqs_deviants_pos = {
               "LOT_repeat" : [0],
               "LOT_alternate": [0,1],
               "LOT_pairs": [0,1,2,3],
               "LOT_quadruplets": [0,3,4,6],
               "LOT_PairsAndAlt1": [3,4,6,7],
               "LOT_Shrinking": [10,11,13,14],
               "LOT_PairsAndAlt2": [3,5,6,7],
               "LOT_threeTwo": [0,2,4,6],
               "LOT_centermirror": [8,11,12,14],
               "LOT_complex": [8,11,13,14],
               # "tree1":[4,7,8,9],
               # "tree2":[4,8,11,12],
               "LocalGlobal_Standard": [4], # Global deviant only on the last element !
               "LocalGlobal_Deviant": [4], # Global deviant only on the last element !

               "LocalGlobal_Omission": [4],  # Global deviant only on the last element !

               "RandReg_5":[1,2,3,4],
               "RandReg_8": [4,5,6,7],
               "RandReg_10":[6,7,8,9],
               "RandReg_20":[16,17,18,19]}

    soa = {
           "LOT_repeat" : 250,
           "LOT_alternate": 250,
           "LOT_pairs": 250,
           "LOT_quadruplets": 250,
           "LOT_PairsAndAlt1": 250,
           "LOT_Shrinking": 250,
           "LOT_PairsAndAlt2": 250,
           "LOT_threeTwo": 250,
           "LOT_centermirror": 250,
           "LOT_complex": 250,
           # "tree1":250,
           # "tree2":250,
           "LocalGlobal_Standard": 150, # Global deviant only on the last element !
           "LocalGlobal_Deviant": 150, # Global deviant only on the last element !

           "LocalGlobal_Omission": 150,  # Global deviant only on the last element !

           "RandReg_5":0,
           "RandReg_8": 0,
           "RandReg_10":0,
           "RandReg_20":0}

    return SeqsInfo(seqs,seqs_number_element,seqs_subsize_add,seqs_inner_repeat,seqs_deviants_pos,soa)

def switch(id, s, nameSeq, switchId,deviants_position,alphabet):
    # id: current position in the regularity to be modified
    # s: sub-sequence at position id
    # switchId: current switchId (we might generate multiple times the different possible switches)
    # deviants_position: indicate for all type of sequences where the deviants should be
    # alphabet: current alphabet of motives in the sequence.

    # The function generate the switch to a deviant condition if id is the current target switch position.
    if id == deviants_position[nameSeq][switchId]:

        # Find the current motif:
        currentMotif = None
        for ida,a in enumerate(alphabet):
            if len(s)==len(a) and np.all(np.equal(s,a)):
                currentMotif = ida
        deviantMotif = np.random.choice(np.setdiff1d(range(len(alphabet)),[currentMotif]),1)[0]

        out = alphabet[deviantMotif]
        if not currentMotif is None:
            if len(alphabet[deviantMotif]) < len(alphabet[currentMotif]):
                to_add = len(alphabet[currentMotif]) - len(alphabet[deviantMotif])
                out = np.concatenate([out,np.random.choice(alphabet[deviantMotif],to_add,replace=True)])
            else:
                out = alphabet[deviantMotif][:len(alphabet[currentMotif])]
        return out
    else:
        return s

class RandReg_structure(SoundGenerator,ElementMasking):

    @classmethod
    def get_info(cls, with_dict=False):

        samplerate = 16000  # 44000 hz in the original study, here using 16000 for wav2vec2

        frequencies = np.logspace(np.log(222), np.log(2000), 20, base=np.exp(1))

        # ## 23/08/2023: We try to see if having more complicated sounds
        # # (here some with multiples frequencies, could change what is happening:
        # frequencies_fundamental = np.logspace(np.log(222),np.log(500),20,base = np.exp(1))
        # frequencies = np.array([[f,f*2,f*4] for f in frequencies_fundamental])

        duration_tone = 50  # ms, duration of the tone
        consine_rmp_length = 5  # ms, duration of the upper and lower cosine ramp

        # Rcyc = [1,5,10] #15,20,40
        # cyc_names = ["block1","block5","block10"]
        Rcyc = [1]
        cyc_names = ["block1"]

        # nb_Rcyc = 8 # NOTE: in the original experiment this is varied between 7 or 8

        if with_dict:
            return {"Rcyc": Rcyc, "frequencies": frequencies, "samplerate": samplerate,
                    "duration_tone": duration_tone, "consine_rmp_length": consine_rmp_length}

        return Rcyc, frequencies, samplerate, duration_tone, consine_rmp_length, cyc_names

    @classmethod
    def pick_sub_sequences(cls,frequencies,seqInfo,seqId,Rcyc):
        subseqs = []
        for j in range(seqInfo.seqs_number_element[seqId]):
            if len(subseqs)>0:
                all_before = np.concatenate(subseqs)
                freq_to_choose_from = np.setdiff1d(frequencies, all_before)
            else:
                freq_to_choose_from = frequencies
            subseqs += [
                np.random.choice(freq_to_choose_from, size=Rcyc + seqInfo.seqs_subsize_add[seqId][j], replace=True)]
        return subseqs

    @classmethod
    def _REG(cls, frequencies, Rcyc, seqInfo: SeqsInfo, seqid,
             number_out_motif=2,
             vary_outer_sequence = True,
             alphabet=None,
             number_inner_repeat = None):

        if number_inner_repeat is None:
            number_inner_repeat = seqInfo.seqs_inner_repeat[seqid]

        if not vary_outer_sequence:
            if alphabet is None:
                alphabet = cls.pick_sub_sequences(frequencies,seqInfo,seqid,Rcyc)
            reg = seqInfo.seqs[seqid](*alphabet)
            reg = duplicateL(reg, number_inner_repeat-1)
            size_reg = np.concatenate(reg).shape[0]

            reg = duplicateL(reg,number_out_motif-1)
            reg = np.concatenate(reg)
        else:
            out = []
            for _ in range(number_out_motif):
                alphabet = cls.pick_sub_sequences(frequencies, seqInfo, seqid, Rcyc)
                reg = seqInfo.seqs[seqid](*alphabet)
                reg = duplicateL(reg, number_inner_repeat - 1)
                size_reg = np.concatenate(reg).shape[0]

                out += reg
            reg = np.concatenate(out)


        return reg,size_reg

    @classmethod
    def _DEVIANT(cls, frequencies, Rcyc, seqInfo : SeqsInfo, seqid,
                 switchId,alphabet=None):
        if alphabet is None:
            alphabet = cls.pick_sub_sequences(frequencies, seqInfo, seqid, Rcyc)
        deviants = seqInfo.seqs[seqid](*alphabet)
        for id in range(len(deviants)):
            try:
                assert seqInfo.seqs_deviants_pos[seqid][switchId]<len(deviants)
            except:
                raise Exception("")
            deviants[id] = switch(id, deviants[id], seqid, switchId, seqInfo.seqs_deviants_pos, alphabet)
        return np.concatenate(deviants)

    @classmethod
    def _REGRAND(cls,frequencies,Rcyc, seqInfo: SeqsInfo, seqid,
                 number_out_motif=2,
                 vary_outer_sequence = True,
                 alphabet = None,
                 sampleRandInAlphabet = False):
        # Rcyc: the size of one cycle
        # size: the total number of Rcyc

        reg,size_reg = cls._REG(frequencies,Rcyc, seqInfo,seqid,
                    number_out_motif=number_out_motif,
                    vary_outer_sequence = vary_outer_sequence,
                    alphabet=alphabet)
        size_rand = size_reg

        if sampleRandInAlphabet:
            rands = np.random.choice(np.concatenate(alphabet), replace=True, size=size_rand)
            while rands[0] == reg[-size_rand]:
                rands = np.random.choice(np.concatenate(alphabet), replace=True, size=size_rand)
        else:
            rands = np.random.choice(frequencies, replace=True, size=size_rand)
        rands_start = np.random.choice(frequencies, replace=True, size=size_rand)
        return np.concatenate([rands_start,reg,rands]), size_rand

    @classmethod
    def _REGDEVIANT(cls,frequencies,Rcyc, seqInfo: SeqsInfo,seqid,
                    number_out_motif=2,
                    vary_outer_sequence = True,
                    alphabet=None):

        # Rcyc: the size of one cycle
        # size: the total number of Rcyc

        regdeviants = []
        if alphabet is None:
            alphabet = cls.pick_sub_sequences(frequencies,seqInfo,seqid,Rcyc)

        if vary_outer_sequence:
            reg_init, size_reg = cls._REG(frequencies, Rcyc, seqInfo, seqid,
                                     number_out_motif = number_out_motif,
                                     vary_outer_sequence=True)
        else:
            reg_init, size_reg = cls._REG(frequencies, Rcyc, seqInfo, seqid,
                                     number_out_motif = number_out_motif,
                                     vary_outer_sequence=False,
                                     alphabet=alphabet)
        rands_start = np.random.choice(frequencies, replace=True, size=size_reg)

        for switchid in range(len(seqInfo.seqs_deviants_pos[seqid])):
            deviant = cls._DEVIANT(frequencies,Rcyc, seqInfo,seqid,
                        switchid,
                        alphabet=alphabet)
            if seqInfo.seqs_inner_repeat[seqid] >1:
                reg,_ = cls._REG(frequencies,Rcyc, seqInfo,seqid,
                            number_out_motif = 1,
                            vary_outer_sequence=False,
                            alphabet=alphabet,
                            number_inner_repeat=seqInfo.seqs_inner_repeat[seqid]  - 1)
                deviant = np.concatenate([reg,deviant])
            try:
                assert len(deviant) == size_reg
            except:
                raise Exception("")

            regdeviants += [[np.concatenate([rands_start,reg_init,deviant]),None]]


        return regdeviants

    @classmethod
    def get_tones(cls,frequencies,samplerate,duration_tone,consine_rmp_length,duration_silence = 0):

        if len(frequencies.shape)==1:
            frequencies = frequencies[:,None]
        tones = np.stack([np.sum(
                np.stack([librosa.tone(f, sr=samplerate, duration=duration_tone / 1000) for f in fs],axis=0), axis=0)
                    for fs in frequencies],axis=0)

        # tones = np.stack([librosa.tone(f, sr=samplerate, duration=duration_tone/1000) for f in frequencies],axis=0)

        # creating ramps
        hanning_window = np.hanning(consine_rmp_length/1000 * samplerate) #5 ms raised cosine window
        hanning_window = hanning_window[:int(np.floor(hanning_window.shape[0] / 2))]
        # filtering tones with ramps:
        tones[:,:hanning_window.shape[0]] = tones[:,:hanning_window.shape[0]] * hanning_window
        tones[:,-hanning_window.shape[0]:] = tones[:,-hanning_window.shape[0]:] * hanning_window[::-1]

        # Pierre 23/08/2023, we normalize, to see if this could help the network to perform better!
        ## Normalization:
        tones = tones / np.sqrt(np.sum(tones ** 2,axis=-1,keepdims=True))

        ## Adds silence if needed:
        if duration_silence>0:
            tones = np.concatenate([tones,np.zeros((tones.shape[0],int(duration_silence/1000*samplerate)))],axis=-1)

        return tones

    @classmethod
    def get_all_seq(cls):
        return get_all_sequences()

    @classmethod
    def generateSounds(cls,dirWav: Union[str,pathlib.Path], dirZarr: Union[str,pathlib.Path]):

        Rcyc,frequencies,samplerate,duration_tone,consine_rmp_length,cyc_names = cls.get_info()

        seqInfo = cls.get_all_seq()
        freq = np.arange(len(frequencies))

        exception = {"block1":["LOT_repeat"]} #{}

        # Block One:
        for blockName,rcyc in zip(cyc_names,Rcyc):
            list_seq = list(seqInfo.seqs.keys())
            if blockName in exception.keys():
                list_seq = np.setdiff1d(list_seq,exception[blockName])
            for seqid in list_seq:

                tones = cls.get_tones(frequencies, samplerate, duration_tone, consine_rmp_length,
                                      duration_silence = seqInfo.soa[seqid])

                tones = np.append(tones, np.zeros((1, len(tones[0])), dtype=np.float32), axis=0)

                for vary_outer_sequence,typeTask in zip([False,True],["Detect","Generalize"]):
                    fileName = blockName+"_"+seqid+"_"+typeTask

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

        Rcyc,frequencies,samplerate,duration_tone,consine_rmp_length,cyc_names = cls.get_info()
        seqInfo = cls.get_all_seq()

        exception = {"block1":["LOT_repeat"]} #{}
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
