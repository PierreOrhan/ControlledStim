import pandas as pd

from sounds.perExperiment.sequences import lot_patterns,ToneList,Sequence,RandomPattern
from sounds.perExperiment.sound_elements import Bip,Silence
from sounds.perExperiment.sound_elements import Sound_pool,Sound, HindiSyllable
from sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from sounds.perExperiment.sound_elements import ramp_sound,normalize_sound
from dataclasses import dataclass,field
import numpy as np
from typing import Union,Tuple
import parselmouth

@dataclass
class Maye2002(Protocol_independentTrial):
    name : str = "Maye2002"
