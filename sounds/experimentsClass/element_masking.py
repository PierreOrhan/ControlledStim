import pandas as pd
from pathlib import Path
from sounds.utils import get_input_lengths
import soundfile as sf
import numpy as np

class ElementMasking:

    name : str
    def __init__(self, name :str):
        self.name = name

    def _mask_and_latent(self, sequence_data_set_dir: str):
        """Preprocessing for self-supervised learning. Masking elements for contrastive learning using wav2vec2.
        Args:
            sequence_data_set_dir (str): Path to the sequence data set directory.
        Returns:
            None
        """
        wav2vec2_receptiveField = 400  # number of input sample that are taken into account in a latent sample
        wav2vec2_stride = 320  # Stride between each latent sample
        wav2vec2_params = {"conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                           "conv_stride": [5, 2, 2, 2, 2, 2, 2]}

        num_negative_samples = 100  # Number of negative samples for contrastive learning

        # Load the sequence data set
        sequences = pd.read_csv(sequence_data_set_dir + "/sequences.csv")
        for sequence in sequences["name","wav_path","sound_info_path"]:
            sequences_info = pd.read_csv(sequence["sound_info_path"])

            sound_mat = sf.read(sequence["wav_path"])

            # Get the number of latent samples
            latent_length = get_input_lengths(np.size(sound_mat),wav2vec2_params["conv_kernel"],wav2vec2_params["conv_stride"])

            # define the intervals of each tone in temporal space
            tone_start = sequences_info["start"]
            tone_duration = sequences_info["duration"]
            tone_end = tone_start + tone_duration
            tone_itv = [pd.Interval(s, e, closed="left") for s, e in zip(tone_start, tone_end)]

            # define the intervals of each tone in latent space
            latentblock_start = np.arange(0, wav2vec2_stride * latent_length, step=wav2vec2_stride)
            latentblock_end = latentblock_start + wav2vec2_receptiveField
            latentblock_interval = [pd.Interval(s, e, closed="left") for s, e in zip(latentblock_start, latentblock_end)]









