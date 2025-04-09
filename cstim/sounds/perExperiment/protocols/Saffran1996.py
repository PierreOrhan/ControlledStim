import pandas as pd
from cstim.sounds.perExperiment.sequences import Sequence
from cstim.sounds.perExperiment.sequences.patterns import WordStream
from cstim.sounds.perExperiment.sound_elements import EnglishSyllable,FrenchSyllable,SoundSegment,Bip

from cstim.sounds.perExperiment.sound_elements import Sound_pool,Sound
from cstim.sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from cstim.sounds.perExperiment.sound_elements import ramp_sound,normalize_sound,Silence
from dataclasses import dataclass,field
import numpy as np
from typing import Union,Tuple,List
from pathlib import Path

@dataclass
class Saffran(Protocol_independentTrial):
    name : str = "Saffran"
    sequence_isi : float = 0
    isi : float = 0
    duration_tone : float = 0.2
    samplerate : int = 16000
    motif_repeat : int = 42
    nb_words : int = 4
    size_words : int = 3
    # tones_fs : Union[list[list[float]],np.ndarray] = field(default_factory=list)
    # s_reg : list[Sound] = field(default=None)

    
    def words_sample(self,syllables):
        ### Sample randomly one of the sequence:
        # In the original experiment the words have no syllable in common so we respect that here.
        words = []
        to_remove = []
        for _ in range(self.nb_words):
            syl = np.random.choice(np.setdiff1d(range(len(syllables)),to_remove),self.size_words,replace=False)
            words += [[syllables[s] for s in syl]]
            to_remove += [syl]
        words = np.array(words)
        return words


    def __post_init__(self):
        self.name = self.name
        syllables = np.array([["t","u"],["p","i"],["r","o"],["b","i"],["d","a"],["k","u"],
                     ["g","o"],["l","a"],["b","u"],["p","a"],["d","o"],["t","i"]])
        self.syllables = ["".join(e) for e in syllables]
    

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        words = self.words_sample(self.syllables)
        sounds_words = [[EnglishSyllable(name=s,syllable=s,samplerate=self.samplerate,duration=self.duration_tone) for s in w] for w in words]
        self.sound_pool = Sound_pool.from_list(np.concatenate(sounds_words))
    
        regSeq = WordStream(nb_words=self.nb_words, size_words=self.size_words, len=self.motif_repeat)        

        all_pool = [self.sound_pool]
        all_seq = [regSeq]
        return all_pool,all_seq

    def _trial(self) -> tuple[list[Sound],int,pd.DataFrame]:
        ''' Trial implements the logic of the protocol for one trial.'''
        all_pool, all_seq = self._getPoolAndSeq()
        all_sound = []
        nb_element = 0
        for p,seq in zip(all_pool, all_seq):
            s_p = seq(p) # combine sequence and pool
            ## Apply sound modifications:
            s_p = [ramp_sound(s,cosine_rmp_length=0.005) for s in s_p]
            all_sound += s_p
            nb_element += np.sum([type(s)!= Silence for s in s_p])
            if self.sequence_isi > 0:
                all_sound += [Silence(samplerate=self.samplerate, duration=self.sequence_isi)]
        # should be a list of Sound
        self.sound_pool.clear_picked()
        ### Further returns additional trial information to be stored:
        df_info = pd.DataFrame.from_dict({"pattern":[all_seq[0].pattern]})
        return (all_sound,nb_element,df_info)



@dataclass
class Saffran_StressClue(Saffran):
    """
        Saffran paradigm with added stress on the first syllable.
    """
    name : str = "Saffran_stressed"

    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        words = self.words_sample(self.syllables)
        sounds_words = [[EnglishSyllable(name=s,syllable=s,samplerate=self.samplerate,duration=self.duration_tone) for s in w] for w in words]
        
        # Generate stress with a slower reading speed and increased duration:
        for w in range(len(sounds_words)):
            sounds_words[w][0] = EnglishSyllable(name=sounds_words[w][0].name,
                                                 syllable=sounds_words[w][0].syllable,samplerate=self.samplerate,duration=2*self.duration_tone,
                                                 speed=0.5*160)
        
        self.sound_pool = Sound_pool.from_list(np.concatenate(sounds_words))
    
        regSeq = WordStream(nb_words=self.nb_words, size_words=self.size_words, len=self.motif_repeat)        

        all_pool = [self.sound_pool]
        all_seq = [regSeq]
        return all_pool,all_seq

@dataclass
class Saffran_otherStim(Saffran):
    """
        Saffran paradigm with different stimulis.
    """
    sound_paths : List[Union[str,Path]] = ""
    start: list[float] = 0
    stop: list[float] = 0.05

    def __post_init__(self):
        self.name = self.name
        syllables =   np.array([["t","u"],["p","i"],["r","o"],["b","i"],["d","a"],["k","u"],
                     ["g","o"],["l","a"],["b","u"],["p","a"],["d","o"],["t","i"]])
        self.syllables = ["".join(e) for e in syllables]

        self.mapping = {s:(self.sound_paths[ide],self.start[ide],self.stop[ide]) for ide,s in enumerate(self.syllables)}


    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        words = self.words_sample(self.syllables)
        sounds_words = [[SoundSegment(name=s,filename=self.mapping[s][0],start=self.mapping[s][1],stop=self.mapping[s][2]) for s in w] for w in words]
        self.sound_pool = Sound_pool.from_list(np.concatenate(sounds_words))
    
        regSeq = WordStream(nb_words=self.nb_words, size_words=self.size_words, len=self.motif_repeat)        

        all_pool = [self.sound_pool]
        all_seq = [regSeq]
        return all_pool,all_seq
    

@dataclass
class Saffran_Tones(Saffran):
    """
        Saffran paradigm with tones stimulis.
    """
    tones : Union[list[float],np.ndarray] = field(default_factory=list)
    
    def __post_init__(self):
        self.name = self.name
        syllables =   np.array([["t","u"],["p","i"],["r","o"],["b","i"],["d","a"],["k","u"],
                     ["g","o"],["l","a"],["b","u"],["p","a"],["d","o"],["t","i"]])
        self.syllables = ["".join(e) for e in syllables]
        self.mapping = {s:[self.tones[ide]] for ide,s in enumerate(self.syllables)}


    def _getPoolAndSeq(self) -> Tuple[list[Sound_pool],list[Sequence]]:
        words = self.words_sample(self.syllables)
        sounds_words = [[Bip(name=s,duration=self.duration_tone,fs=self.mapping[s]) for s in w] for w in words]
        self.sound_pool = Sound_pool.from_list(np.concatenate(sounds_words))
    
        regSeq = WordStream(nb_words=self.nb_words, size_words=self.size_words, len=self.motif_repeat)        
        all_pool = [self.sound_pool]
        all_seq = [regSeq]
        return all_pool,all_seq