import numpy as np

from sounds.perExperiment.sequences import FullCommunityGraph
from sounds.perExperiment.sound_elements import FrenchSyllable,Bip,Silence
from sounds.perExperiment.sound_elements import Sound_pool,Sound
from sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from sounds.perExperiment.sound_elements import ramp_sound,normalize_sound
from dataclasses import dataclass,field
import pandas as pd
from typing import Union

@dataclass
class Benjamin2023(Protocol_independentTrial):
    name : str = "HigherGraph"
    duration_tone : float = 0.275
    samplerate : int = 16000
    isi : float = 0.0
    walk_length : int = 960
    tones_fs : Union[list[float],np.ndarray] = field(default=np.linspace(300,1800,num=12))

    def __post_init__(self):
        sounds = [Bip(samplerate=self.samplerate,duration=self.duration_tone,fs=[f]) for f in self.tones_fs]
        self.sound_pool = Sound_pool.from_list(sounds)
        self.seq = FullCommunityGraph(isi=self.isi,walk_length=self.walk_length)

    def _trial(self) -> tuple[list[Sound],int,pd.DataFrame]:
        ''' Trial implements the logic of the protocol for one trial.'''

        ## Instantiate the vocabularies:
        all_pool = [self.sound_pool]
        all_seq = [self.seq]
        all_sound = []
        nb_element = 0
        for p,seq in zip(all_pool, all_seq):
            s_p = seq(p) # combine sequence and pool
            ## Apply sound modifications:
            s_p = [normalize_sound(ramp_sound(s,cosine_rmp_length=0.02)) for s in s_p]
            all_sound += s_p
            nb_element += np.sum([type(s)!= Silence for s in s_p])

        return (all_sound,nb_element,pd.DataFrame.from_dict({"duration_tone":self.duration_tone,"isi":self.isi,
                                                             "walk_length":self.walk_length}))

@dataclass
class Benjamin2023_syllable(Benjamin2023):
    name : str = "HigherGraph_syllable"
    def __post_init__(self):
        all_syllables = np.array([["t", "u"], ["p", "i"], ["r", "o"], ["b", "i"], ["d", "a"], ["k", "u"],
                                  ["g", "o"], ["l", "a"], ["b", "u"], ["p", "a"], ["d", "o"], ["t", "i"]])
        all_syllables = ["".join(a) for a in all_syllables]
        sounds = [FrenchSyllable(samplerate=self.samplerate,duration=self.duration_tone,syllable=a)
                  for a in all_syllables]
        self.sound_pool = Sound_pool.from_list(sounds)
        self.seq = FullCommunityGraph(isi=self.isi,walk_length=self.walk_length)

        self.tones_fs = None # should not be used...