from sounds.perExperiment.sequences import lot_patterns,ToneList,Sequence,RandomPattern
from sounds.perExperiment.sound_elements import Bip,Silence
from sounds.perExperiment.sound_elements import Sound_pool,Sound
from sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from sounds.perExperiment.sound_elements import ramp_sound,normalize_sound
from dataclasses import dataclass,field
import numpy as np
from typing import Union,Tuple

@dataclass
class RandRegRand_LOT(Protocol_independentTrial):
    name : str = "RandRegRand_LOT"
    sequence_isi : float = 0.3
    isi : float = 0.2
    duration_tone : float = 0.05
    samplerate : int = 16000
    motif_repeat : int = 3
    lot_seq : str = "pairs"
    tones_fs : Union[list[float],np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        self.name = self.name+"_"+self.lot_seq
        sounds = [Bip(name="bip-" + str(idf), samplerate=self.samplerate, duration=self.duration_tone, fs=[f]) for
                  idf, f in enumerate(self.tones_fs)]
        # Note: naming the bip is useful to one who is where.
        self.sound_pool = Sound_pool.from_list(sounds)
        self.randSeq = ToneList(isi=self.isi, cycle=16)
        self.regSeq = lot_patterns[self.lot_seq](isi=self.isi)
        self.randSeqEnd = RandomPattern(isi=self.isi,nb_unique_elements=2,len=16)
        ## Make sure the first random tone breaks the sequence:
        if self.randSeqEnd.pattern[0] == self.regSeq.pattern[0]:
            self.randSeqEnd.pattern[0] = 1-self.randSeqEnd.pattern[0]

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        ## Instantiate the vocabularies:
        s_rand = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(16))
        s_reg = Sound_pool.from_list(s_rand.pick_norepeat_n(2))

        all_pool = [s_rand] + [s_reg for _ in range(self.motif_repeat)] + [s_reg]
        all_seq = [self.randSeq] + [self.regSeq for _ in range(self.motif_repeat)] + [self.randSeqEnd]
        return all_pool,all_seq
    def _trial(self) -> tuple[list[Sound],int]:
        ''' Trial implements the logic of the protocol for one trial.'''
        all_pool, all_seq = self._getPoolAndSeq()
        all_sound = []
        nb_element = 0
        for p,seq in zip(all_pool, all_seq):
            s_p = seq(p) # combine sequence and pool
            ## Apply sound modifications:
            for s in s_p:
                ramp_sound(s)
                normalize_sound(s)
            all_sound += s_p
            nb_element += len(s_p)
            if self.sequence_isi > 0:
                all_sound += [Silence(samplerate=self.samplerate, duration=self.sequence_isi)]
        # should be a list of Sound
        self.sound_pool.clear_picked()
        return (all_sound,nb_element)

@dataclass
class RandRegRand_LOT_deviant(RandRegRand_LOT):
    deviant_pos : int = 0
    def __post_init__(self):
        sounds = [Bip(name="bip-"+str(idf),samplerate=self.samplerate, duration=self.duration_tone, fs=[f]) for idf,f in enumerate(self.tones_fs)]
        # Note: naming the bip is useful to one who is where.
        self.sound_pool = Sound_pool.from_list(sounds)
        self.randSeq = ToneList(isi=self.isi, cycle=16)
        self.regSeq = lot_patterns[self.lot_seq](isi=self.isi)

        self.devSeq = lot_patterns[self.lot_seq](isi=self.isi)
        self.devSeq.as_deviant_pattern(self.deviant_pos)

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool], list[Sequence]]:
        ## Instantiate the vocabularies:
        s_rand = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(16))
        s_reg = Sound_pool.from_list(s_rand.pick_norepeat_n(2))

        all_pool = [s_rand] + [s_reg for _ in range(self.motif_repeat)] + [s_reg]
        all_seq = [self.randSeq] + [self.regSeq for _ in range(self.motif_repeat)] + [self.devSeq]
        return all_pool, all_seq

@dataclass
class RandRegRand_LOT_Generalize(RandRegRand_LOT):
    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        ## Instantiate the vocabularies:
        ## In this case we want to change the tone used in the generalize sequence at every step
        # so we pick enough tone in a pool and probably forbid to take them...
        s_poolReg = Sound_pool.from_list(self.sound_pool.pick_norepeat_n(self.motif_repeat*2))
        s_rand = Sound_pool.from_list(s_poolReg.pick_norepeat_n(16))
        s_poolReg.clear_picked() # clear the poolReg to be able to choose again from the self.motif_repeat*2
        s_regs = [Sound_pool.from_list(s_poolReg.pick_norepeat_n(2)) for _ in range(self.motif_repeat)]
        all_pool = [s_rand] + s_regs + [s_regs[-1]]
        all_seq = [self.randSeq] + [self.regSeq for _ in range(self.motif_repeat)] + [self.randSeqEnd]
        return all_pool,all_seq