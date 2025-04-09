import numpy as np
import pandas as pd

from cstim.sounds.perExperiment.sequences import ToneList
from cstim.sounds.perExperiment.sound_elements import Bip,Silence,EnglishSyllable
from cstim.sounds.perExperiment.sound_elements import Sound_pool,Sound
from cstim.sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from cstim.sounds.perExperiment.sound_elements import ramp_sound,normalize_sound
from dataclasses import dataclass,field

from typing import Union

@dataclass
class RandRegRand(Protocol_independentTrial):
    name : str = "RandRegRand"
    sequence_isi : float = 0.150
    cycle : int = 5
    duration_tone : float = 0.05
    samplerate : int = 16000
    isi : float = 0.0
    rand_voc : int = 20
    motif_repeat : int = 3
    tones_fs : Union[list[float],np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        sounds = [Bip(name="Bip-"+str(idf),samplerate=self.samplerate,duration=self.duration_tone,fs=[f]) for idf,f in enumerate(self.tones_fs)]
        self.sound_pool = Sound_pool.from_list(sounds)
        self.seqRand = ToneList(isi=self.isi,cycle=self.rand_voc)
        self.seq = ToneList(isi=self.isi, cycle=self.cycle)

    def _trial(self) -> tuple[list[Sound],int,pd.DataFrame]:
        ''' Trial implements the logic of the protocol for one trial.'''

        ## Instantiate the vocabularies:
        s_rand = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.rand_voc))
        s_reg = Sound_pool.from_list(s_rand.pick_norepeat_n(self.cycle))
        s_randEnd = Sound_pool.from_list(s_reg.pick_norepeat_n(self.cycle))
        ## Make sure the first random tone breaks the sequence:
        if s_randEnd[0] ==s_reg[-1]:
            c=s_randEnd[1]
            s_randEnd[1]=s_randEnd[0]
            s_randEnd[0] = c

        all_pool = [s_rand]+[s_reg for _ in range(self.motif_repeat)] + [s_randEnd]
        all_seq = [self.seqRand] + [self.seq for _ in range(self.motif_repeat)] + [self.seq]

        all_sound = []
        nb_element = 0
        for p,seq in zip(all_pool, all_seq):
            s_p = seq(p) # combine sequence and pool
            ## Apply sound modifications:
            s_p = [normalize_sound(ramp_sound(s)) for s in s_p]
            all_sound += s_p
            nb_element += np.sum([type(s)!= Silence for s in s_p])
            if self.sequence_isi > 0:
                all_sound += [Silence(samplerate=self.samplerate, duration=self.sequence_isi)]
        # should be a list of Sound
        self.sound_pool.clear_picked()
        return (all_sound,nb_element,pd.DataFrame.from_dict({"cycle":[self.cycle],
                                                             "sequence_isi":[self.sequence_isi],
                                                             "isi":[self.isi],
                                                             "motif_repeat":[self.motif_repeat]}))


from cstim.sounds.perExperiment.sound_elements.segment_elements import SoundSegment
from typing import List
from pathlib import Path

@dataclass
class RandRegRand_otherStim(RandRegRand):
    """
        RandReg paradigm with different stimulis.
    """
    sound_paths : List[Union[str,Path]] = ""
    start: list[float] = 0
    stop: list[float] = 0.05
    def __post_init__(self):
        super().__post_init__()
        sounds = [SoundSegment(name="bip-" + str(idf), 
                               filename = self.sound_paths[idf],
                               start = self.start[idf],
                               stop = self.stop[idf]) for idf in range(len(self.sound_paths))]
        # Note: naming the bip is useful to know who is where.
        self.sound_pool = Sound_pool.from_list(sounds)

@dataclass
class RandRegRand_syllable(RandRegRand):
    """
        RandReg paradigm with syllable stimulis.
    """
    
    def __post_init__(self):
        syllables =   np.array([["t","u"],["p","i"],["r","o"],["b","i"],["d","a"],["k","u"],
                     ["g","o"],["l","a"],["b","u"],["p","a"],["d","o"],["t","i"]])
        self.syllables = ["".join(e) for e in syllables]
        self.rand_voc = len(self.syllables)
        super().__post_init__()


        sounds = [EnglishSyllable(name="syllable-" + str(ids), 
                                    syllable=s,samplerate=self.samplerate,duration=self.duration_tone)
                                    for ids,s in enumerate(self.syllables)]
        self.sound_pool = Sound_pool.from_list(sounds)