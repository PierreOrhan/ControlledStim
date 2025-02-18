import numpy as np
import pandas as pd

from sounds.perExperiment.sequences import ToneList
from sounds.perExperiment.sound_elements import Bip,Silence
from sounds.perExperiment.sound_elements import Sound_pool,Sound
from sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from sounds.perExperiment.sound_elements import ramp_sound,normalize_sound
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
        if s_randEnd[0].first_freq==s_reg[-1].first_freq:
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