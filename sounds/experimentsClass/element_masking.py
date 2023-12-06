import pandas as pd
from pathlib import Path
from sounds.utils import get_input_lengths
import soundfile as sf
import numpy as np

from datasets import Dataset

class ElementMasking:

    name : str
    def __init__(self, name :str):
        self.name = name

    def _mask_and_latent(self, sequence_data_set_dir: str, oneEvalPerEvent: bool = True) -> Dataset:
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
        time = 0
        complete_tone_itv = []
        complete_latentblock_itv = []
        max_latent_length = 0
        max_sounds_per_sequence = 0
        for i in range(sequences.shape[0]):
            sequence = sequences.iloc[i, :]
            sequence_info = pd.read_csv(sequence["sound_info_path"])
            max_sounds_per_sequence = max(max_sounds_per_sequence,sequence_info.shape[0])

            sound_mat, sr = sf.read(sequence["wav_path"])

            # Get the number of latent samples
            latent_length = get_input_lengths(len(sound_mat),wav2vec2_params["conv_kernel"],wav2vec2_params["conv_stride"])
            max_latent_length = max(max_latent_length,latent_length)

            # define the intervals of each tone in temporal space
            tone_start = sequence_info["start"] + time
            tone_duration = sequence_info["duration"]
            tone_end = tone_start + tone_duration

            # define the intervals of each tone in latent space
            latentblock_start = np.arange(time*sr, wav2vec2_stride * latent_length, step=wav2vec2_stride)
            latentblock_end = latentblock_start + wav2vec2_receptiveField
            complete_latentblock_itv += [pd.Interval(s, e, closed="left") for s, e in zip(latentblock_start, latentblock_end)]
            toneStart_sample = np.array(tone_start * 16000, dtype=int)
            toneEnd_sample = np.array(tone_end * 16000, dtype=int)
            complete_tone_itv += [pd.Interval(s, e, closed="left") for s, e in zip(toneStart_sample, toneEnd_sample)]

            time += sequence["duration"]


        # Find the block inside with overlap with the tones
        tone_in_block = np.array([[ti.overlaps(lti) for lti in complete_latentblock_itv] for ti in complete_tone_itv])

        block_inside_tone = np.array(
            [[ti.left <= lti.left and ti.right >= lti.right for lti in complete_latentblock_itv] for ti in complete_tone_itv])
        assert np.all(np.any(block_inside_tone, axis=-1))
        # A block that is fully contained in the tone
        ltrs = np.stack([block_inside_tone for _ in range(sequences.shape[0])], axis=0) # latent time reduction blocks
        ## stack: we have the same structure for all the sounds here (one type of sequence), so the
        # focus of the loss will be on the same latent.

        mask_time_indices = np.zeros((sequences.shape[0], max_sounds_per_sequence, max_latent_length), dtype=bool)
        sampled_negative_indices = np.zeros((sequences.shape[0], max_sounds_per_sequence, max_latent_length,
                                                 num_negative_samples), dtype=int)


        all_tones = []
        for seq_path in sequences["wav_path"]:
            seq = pd.read_csv(seq_path)
            all_tones += [seq["name"]]
        all_tones = np.array(all_tones)

        id_seq = 0
        for seq_path in sequences["wav_path"]:
            seq = pd.read_csv(seq_path)
            toneType = seq["name"].to_numpy()
            negative_dic = {}
            for tt in np.unique(all_tones):
                is_same_tone = np.array([tt == t for t in toneType])
                ok_block = np.any(tone_in_block[np.logical_not(is_same_tone)], axis=0) * np.all(
                    np.logical_not(tone_in_block[is_same_tone]), axis=0)
                try:
                    id_ok = np.random.choice(np.where(ok_block)[0], num_negative_samples, replace=False)
                except:
                    print("attention, less than 100 possible negatives, negatives might be too similar between "
                          "each other to give comparable loss values")
                    try:
                        id_ok = np.random.choice(np.where(ok_block)[0], num_negative_samples, replace=True)
                    except:
                        raise Exception("")
                negative_dic[tt] = id_ok

            negative_masks = []
            for toneblock, tt in zip(tone_in_block, toneType):
                negative_mask = np.zeros((max_latent_length, num_negative_samples), dtype=int)
                for i in np.where(toneblock)[0]:
                    negative_mask[i, :] = negative_dic[tt]
                negative_masks += [negative_mask]
            mat_negative_mask = np.stack(negative_masks, axis=0)

            mask_time_indices[id_seq, :, :] = tone_in_block
            sampled_negative_indices[id_seq, :, :, :] = mat_negative_mask
            id_seq += 1

            dataset = Dataset.from_dict({"mask_time_indices": mask_time_indices,
                                         "sampled_negative_indices": sampled_negative_indices})

            return dataset










