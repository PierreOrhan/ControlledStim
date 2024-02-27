import pandas as pd

from sounds.perExperiment.sequences import lot_patterns, ToneList, Sequence, RandomPattern
from sounds.perExperiment.sound_elements import Bip, Silence
from sounds.perExperiment.sound_elements import Sound_pool, Sound, HindiSyllable
from sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from sounds.perExperiment.sound_elements import ramp_sound, normalize_sound
from dataclasses import dataclass, field
import numpy as np
from typing import Union, Tuple
import parselmouth


@dataclass
class Maye2002(Protocol_independentTrial):
    name: str = "Maye2002"
    duration_tone: float = 0.275
    samplerate: int = 16000
    isi: float = 0.0

    def __post_init__(self):
        self.syllables: np.array = self.generate_syllables()

    def generate_syllables(self, VOT_cont: int = 4, VOT_step: float = 0.025):
        pass

    def _trial(self) -> tuple[list[Sound], int, pd.DataFrame]:
        """ Trial implements the logic of the protocol for one trial."""
