from abc import ABC

import numpy as np

from sounds.perExperiment.sequences.patterns import SyllableTriplet
from sounds.perExperiment.sound_elements.speech_elements import FrenchSyllable
from sounds.perExperiment.sound_elements.tones_elements import Bip
from sounds.perExperiment.sound_elements import Sound_pool,Sound,Silence
from sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from sounds.perExperiment.sound_elements import ramp_sound,pitch_shift
from dataclasses import dataclass
import pandas as pd
import copy
from typing import Union
import librosa
import torch


@dataclass
class PitchRuleDeviant_1(Protocol_independentTrial):
    name : str = "PitchRuleDeviant_1"
    duration_tone : float = 0.250
    samplerate : int = 16000
    sequence_isi : float = 0.700
    isi : float = 0.050
    nb_standard : int = 66
    nb_deviant_pitch : int = 8
    nb_deviant_rule : int = 8
    cycle : int = 3

    def __post_init__(self):
        A_syllables = np.array([["f", "i"], ["l", "e"]])
        B_syllables = np.array([["t", "o"], ["b", "u"]])
        X_syllables = np.array([["k", "a"], ["w", "e"], ["m", "i"], ["n", "o"], ["g", "u"], ["s", "a"],
                                  ["m", "e"], ["r", "i"], ["r", "o"], ["k", "u"], ["m", "a"], ["k", "e"],
                                  ["g", "i"], ["k", "o"], ["s", "u"], ["w", "a"], ["x", "e"], ["k", "i"],
                                  ["s", "o"], ["m", "u"]])
        A_syllables = ["".join(a) for a in A_syllables]
        B_syllables = ["".join(a) for a in B_syllables]
        X_syllables = ["".join(a) for a in X_syllables]
        all_syllables = A_syllables + B_syllables + X_syllables
        # AXB is the sound structure.
        # sounds = [EnglishSyllable(samplerate=self.samplerate, duration=self.duration_tone, syllable=a)
        #           for a in all_syllables]
        sounds = [FrenchSyllable(samplerate=self.samplerate, syllable=a,
                                 force_duration=False,
                                 pitch_modifiers=[],voice_id=2) for a in all_syllables]
        # sounds = [Bip(samplerate=self.samplerate, duration=self.duration_tone, fs=[300, 800]) for a in all_syllables]
        self.sound_pool = Sound_pool.from_list(sounds)
        self.seq = SyllableTriplet(isi=self.isi)

    def _getPoolAndSeq(self):

        ## Instantiate the vocabularies:
        all_pool = []

        for i in range(self.nb_standard):
            i_s1 = np.random.choice(2)
            i_s2 = np.random.choice(20)
            s = [self.sound_pool[i_s1],self.sound_pool[i_s2],self.sound_pool[i_s1+2]]
            # fi is always with to and le is always with bu
            all_pool.append(Sound_pool.from_list(s))

        for i in range(self.nb_deviant_rule):
            i_s1 = np.random.choice(2)
            i_s2 = np.random.choice(20)
            if i_s1 == 0:
                s = [self.sound_pool[i_s1],self.sound_pool[i_s2],self.sound_pool[i_s1+3]]
                # deviant rule: fi is with bu and le with to
            else:
                s = [self.sound_pool[i_s1],self.sound_pool[i_s2],self.sound_pool[i_s1+1]]
                # second deviant rule
            all_pool.append(Sound_pool.from_list(s))

        for i in range(self.nb_deviant_pitch):
            i_s1 = np.random.choice(2)
            i_s2 = np.random.choice(20)
            s = [self.sound_pool[i_s1],self.sound_pool[i_s2],self.sound_pool[i_s1+2]]
            for j in range(len(s)):
                newS = pitch_shift(s[j],ratio_pitch = 1.11)
                newS.name += "_shifted"
                s[j] = newS
                ## double check this part.
            all_pool.append(Sound_pool.from_list(s))

        all_pool = np.random.permutation(all_pool)
        all_seq = [self.seq for _ in range(len(all_pool))]
        return all_pool,all_seq

    def _trial(self) -> tuple[list[Sound],int,pd.DataFrame]:
        ''' Trial implements the logic of the protocol for one trial.'''

        all_pool, all_seq = self._getPoolAndSeq()

        all_sound = []
        nb_element = 0
        for p,seq in zip(all_pool, all_seq):
            s_p = seq(p) # combine sequence and pool
            ## Apply sound modifications:
            s_p = [ramp_sound(s,cosine_rmp_length=0.02) for s in s_p]
            all_sound += s_p
            nb_element += np.sum([type(s) != Silence for s in s_p])
            if self.sequence_isi > 0:
                all_sound += [Silence(samplerate=self.samplerate, duration=self.sequence_isi)]

        dict_data = {"cycle":self.cycle,"sequence_isi":self.sequence_isi,"isi":self.isi,
                     "nb_standards":self.nb_standard,
                     "nb_deviant_pitch":self.nb_deviant_pitch,
                     "nb_deviant_rule":self.nb_deviant_rule}
        df = pd.DataFrame.from_dict(dict_data, orient='index', columns=['values'], dtype=None)
        df = df.transpose()
        return (all_sound,nb_element,df)
