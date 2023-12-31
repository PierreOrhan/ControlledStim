import os.path
from typing import Union
import numpy as np
import soundfile as sf
import pandas as pd
from sounds.perExperiment.sound_elements import Sound
from dataclasses import dataclass,field
from pathlib import Path
from abc import abstractmethod,ABC

class Protocol(ABC):
    @abstractmethod
    def __init__(self,*args):
        """ The init fonction should be used to define the different properties of the protocol """
        pass
    @abstractmethod
    def generate(self,*args) -> pd.DataFrame:
        """ The generate method should be the method to create all sounds of the final dataset, as .wav file.
            Additionnaly, each file should be accompagnied by a .csv file indicating sound events in the .wav file.
            Finally a trials.csv file summarizes the whole dataset"""
        pass

class Protocol_independentTrial(Protocol):
    name : str = field(default=str)
    samplerate: int = 16000
    @abstractmethod
    def _trial(self) -> tuple[list[Sound],int]:
        raise Exception("method to subclass")

    def generate(self,n_trial : int ,output_dir : Union[str,Path]) -> pd.DataFrame:
        """
        Generates a dataset according to a protocol. Each trial is independently
        generated through a call to _trial, which user have to subclass.
        The trial consists of a sound waveform, annotated with all the sound elements in the waveform.
        These annotations are stored in a .csv file along with the stimuli (one folder for .wav and one folder for .csv)

        :param n_trial: the number of trial to generate independently
        :param output_dir: the directory where the dataset is generated
        :return: a trials.csv dataFrame which describe the dataset.
        """
        name_trials,wav_paths,mask_info_path,sound_durations,sound_info_paths,number_elements = [],[],[],[],[],[]
        for ntrial in range(n_trial):
            name = self.name+"_trial-"+str(ntrial)
            all_sound,nb_element = self._trial()
            sd = [s.sound for s in all_sound]
            names = [s.name for s in all_sound]

            # save the sound
            sd_out = np.concatenate(sd)
            os.makedirs(output_dir / "sounds", exist_ok=True)
            sf.write(Path(output_dir) / "sounds" /  (name + ".wav"), sd_out, samplerate=self.samplerate)

            ## Pierre: the advantage of a having a list of Sound would be to be able to access
            # all the information of the Sound, including .duration here. It is bad to have to use a dictionnary
            durations = [s.duration for s in all_sound]
            start = np.cumsum([0] + durations[:-1])
            df_sound_info = pd.DataFrame()
            df_sound_info["name"] = names
            df_sound_info["start"] = np.array(start)
            df_sound_info["duration"] = durations
            os.makedirs(output_dir / "sound_info", exist_ok=True)
            df_sound_info.to_csv(Path(output_dir) / "sound_info" /  (name + ".csv"), index=False)

            name_trials += [name]
            wav_paths += [str(Path(output_dir) /  (name + ".wav"))]
            mask_info_path += [None]
            sound_durations += [sd_out.shape[0]]
            sound_info_paths += [str(Path(output_dir) / "sound_info" /  (name + ".csv"))]
            number_elements += [nb_element]

        # generating a csv with the sequence names and the corresponding soundfile paths
        df = pd.DataFrame()
        df["name"] = name_trials
        df["wav_path"] = wav_paths
        df["mask_info_path"] = mask_info_path
        df["duration"] = sound_durations
        df["sound_info_path"] = sound_info_paths
        df["number_element"] = number_elements
        df.to_csv(Path(output_dir) / "trials.csv", index=False)
        return df

# class Protocol:
#     def __init__(self, name: str, sequence_isi: float):
#         self.name = name
#         self.sequence_isi = sequence_isi
#
#     def create(self, soundseq_dataset_csv: str):
#         df = pd.read_csv(soundseq_dataset_csv, index_col=False)
#         ann_dict = {
#             "audio_data": [],
#             "sample_rate": [],
#             "label": []
#         }
#         for seq in protocol_name_to_sequences[self.name]:
#             data, sr = sf.read(df[df["name"] == seq]["wav_path"].values[0])
#             ann_dict["audio_data"].append(data)
#             ann_dict["sample_rate"].append(sr)
#             ann_dict["label"].append(seq)
#
#         ann_dataset = Dataset.from_dict(ann_dict)
#         return ann_dataset