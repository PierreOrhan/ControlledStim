import librosa
import numpy as np
import pandas as pd
from sounds.perExperiment.sequences import ToneList
from sounds.perExperiment.sound_elements import Silence,FrenchSyllable
from sounds.perExperiment.sound_elements import Sound_pool,Sound
from sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from sounds.perExperiment.sound_elements import ramp_sound,normalize_sound
from dataclasses import dataclass,field

from typing import Union,Tuple

@dataclass
class AXCSyllableStream(Protocol_independentTrial):
    name : str = "AXCSyllableStream"
    sequence_isi : float = 0.0
    duration_tone : float = 0.262
    samplerate : int = 16000
    isi : float = 0.0
    rand_voc : int = 20
    motif_repeat : int = 3
    tones_fs : Union[list[float],np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        ## Remark: TODO: the author puts a progressive fade-in and fade-out of the stream
        ## Remark: the authors have two additional controls over this experiment.

        # AXC stimulis:
        family1 = [["pu","li","ki"],["pu","ra","ki"],["pu","fo","ki"]]
        family2 = [["be","li","ga"],["be","ra","ga"],["be","fo","ga"]]
        family3 = [["ta","li","du"],["ta","ra","du"],["ta","fo","du"]]

        self.familyPool = []
        for fam in [family1,family2,family3]:
            sounds = [[FrenchSyllable(samplerate=self.samplerate,duration=self.duration_tone,
                                     syllable=p,voice_id=2
                                      # pitch_modifiers=[(100.0,200.0)]
                                      ,force_duration=False) for p in f] for f in fam]

            # Pitch modifiers for the totality of the sound, targeting a pitch of 200Hz
            self.familyPool += [sounds]

    def _familiarization_stream(self) -> tuple[list[Sound],int]:
        # Familiarizations stream in experiment 1 and 2 of pena et al. 2002 Science
        # Constraint:
        # 1) a word of a family is not followed by another word of the same family
        # 2) two words are not adjacent if they have the same intermediate syllable
        # 100 repetition of each word in the steam.
        # Ramp of 5 seconds in onset and offset
        # words: mean lenght of 696 ms; syllable duration of 232 ms.
        # pitch: 200Hz

        total_nb_words = 900
        samples = [0]
        for _ in range(total_nb_words-1):
            samples += [np.random.choice(np.setdiff1d(range(3),[samples[-1]]),1)[0]]
        family_sequences = np.array(samples) # sequence of family, verify criteria 1

        word_samples = [0]
        for _ in range(total_nb_words-1):
            word_samples += [np.random.choice(np.setdiff1d(range(3),[word_samples[-1]]),1)[0]]
        word_samples = np.array(word_samples) # sequence of word, verify criteria 2

        all_sound = []
        nb_element = 0
        for f,w in zip(family_sequences, word_samples):
            s_p = self.familyPool[f][w]
            ## Apply sound modifications:
            s_p = [ramp_sound(s,cosine_rmp_length=0.1) for s in s_p]
            all_sound += s_p
            nb_element += np.sum([type(s)!= Silence for s in s_p])
            if self.sequence_isi > 0:
                all_sound += [Silence(samplerate=self.samplerate, duration=self.sequence_isi)]
        return all_sound,nb_element

    def _trial(self) -> tuple[list[Sound],int,pd.DataFrame]:
        ''' Trial implements the logic of the protocol for one trial.'''
        all_sound,nb_element = self._familiarization_stream()
        return (all_sound,nb_element,pd.DataFrame.from_dict({"sequence_isi":self.sequence_isi
                                                             ,"isi":self.isi}))


@dataclass
class AXCSyllableStream_exp1(AXCSyllableStream):

    def __post_init__(self):
        super(AXCSyllableStream_exp1,self).__post_init__()
        ## Parts words:
        #CAX part words:
        pw1 = [["ki","ta","ra"],["ki","ta","fo"],["ga","pu","fo"],["du","be","ra"]]
        # XCA part-words:
        pw2 = [["li","ki","ta"],["ra","du","be"],["ra","ga","pu"],["fo","ga","pu"]]

        self.partWordPool = []
        for pw in pw1+pw2:
            sounds = [FrenchSyllable(samplerate=self.samplerate,duration=self.duration_tone,
                                     syllable=p,voice_id=2,
                                      pitch_modifiers=[],
                                     force_duration=False) for p in pw]
            # Pitch modifiers for the totality of the sound, targeting a pitch of 200Hz
            self.partWordPool += [sounds]


    def _test_stream(self) -> tuple[list[Sound],int]:
        # 36 pairs each consisting of a word and a part word:
        pws = np.random.choice(range(len(self.partWordPool)),36,replace=True)
        wordsFams = np.random.choice(range(3),36,replace=True)
        words = np.random.choice(range(3),36,replace=True)
        sounds = []
        nb_elements = 0
        for pw,wf,w in zip(pws,wordsFams,words):
            sounds += [ramp_sound(s,cosine_rmp_length=0.005) for s in self.partWordPool[pw]]
            sounds += [Silence(duration=0.5,samplerate=self.samplerate)]
            nb_elements += 3
            sounds += [ramp_sound(s,cosine_rmp_length=0.005) for s in self.familyPool[wf][w]]
            sounds += [Silence(duration=2,samplerate=self.samplerate)]
            nb_elements += 3
        return sounds,nb_elements

    def _trial(self) -> tuple[list[Sound],int,pd.DataFrame]:
        ''' Trial implements the logic of the protocol for one trial.'''
        all_sound,nb_element = self._familiarization_stream()
        all_soundtest,nb_elementtest = self._test_stream()

        return (all_sound+all_soundtest,nb_element+nb_elementtest
                ,pd.DataFrame.from_dict({"sequence_isi":[self.sequence_isi],"isi":[self.isi]}))