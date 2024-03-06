from typing import Union
import pathlib
from abc import ABC,abstractmethod
from datasets import Dataset

class SoundGenerator(ABC):
    @abstractmethod
    def generateSounds(self,
                       dirWav: Union[str,pathlib.Path],
                       dirDataset: Union[str,pathlib.Path]) -> Dataset:
        # Generates a set of experimental stimulis.
        #   Wav files are put in dirWav.
        #   Waveform are also put in the dirZarr directory but as a zarr groups.
        #   Other needed files can be added into the dirZarr directory as zarr groups
        pass