import pandas as pd
from pathlib import Path
from sounds.utils import get_input_lengths
import soundfile as sf
import numpy as np

class ElementMasking:

    name : str
    def __init__(self, name :str):
        self.name = name

    def _mask_and_latent(self, sequence_data_set_dir: str, oneEvalPerEvent: bool = True):
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
        for sequence in sequences["name","wav_path","sound_info_path"]:
            sequence_info = pd.read_csv(sequence["sound_info_path"])

            sound_mat = sf.read(sequence["wav_path"])

            # Get the number of latent samples
            latent_length = get_input_lengths(np.size(sound_mat),wav2vec2_params["conv_kernel"],wav2vec2_params["conv_stride"])

            # define the intervals of each tone in temporal space
            tone_start = sequence_info["start"] + time
            tone_duration = sequence_info["duration"]
            tone_end = tone_start + tone_duration

            # define the intervals of each tone in latent space
            latentblock_start = np.arange(time, wav2vec2_stride * latent_length, step=wav2vec2_stride)
            latentblock_end = latentblock_start + wav2vec2_receptiveField
            complete_latentblock_itv += [pd.Interval(s, e, closed="left") for s, e in zip(latentblock_start, latentblock_end)]
            toneStart_sample = np.array(tone_start * 16000, dtype=int)
            toneEnd_sample = np.array(tone_end * 16000, dtype=int)
            complete_tone_itv += [pd.Interval(s, e, closed="left") for s, e in zip(toneStart_sample, toneEnd_sample)]

            time += sequence["duration"]


        # Find the block inside with overlap with the tones
        tone_in_block = np.array([[ti.overlaps(lti) for lti in complete_latentblock_itv] for ti in complete_tone_itv])

        block_inside_tone = np.array(
            [[ti.left <= lti.left and ti.right >= lti.right for lti in latentblock_interval] for ti in tone_itv])
        assert np.all(np.any(block_inside_tone, axis=-1))
        # A block that is fully contained in the tone
        ltrs = np.stack([block_inside_tone for _ in range(sequence_info.shape[0])], axis=0)
        ## stack: we have the same structure for all the sounds here (one type of sequence), so the
        # focus of the loss will be on the same latent.

        if "mask_time_indices" not in sequence_info.keys():
            mask_time_indices = np.zeros((sequence_info.shape[0], latent_length), dtype=bool)
            sampled_negative_indices = np.zeros((sequence_info.shape[0], latent_length,
                                                     num_negative_samples), dtype=int)
            sequence_info["latent_time_reduction"] = ltrs
        else:
            # TODO
            pass

        all_tones = []
        for seq_path in sequence_info["wav_path"]:
            seq = pd.read_csv(seq_path)
            all_tones += [seq["name"]]
        all_tones = np.array(all_tones)

        for seq_path in sequence_info["wav_path"]:
            id_sound = 0
            seq = pd.read_csv(seq_path)
            toneType = seq["name"].to_numpy()
            negative_dic = {}
            for tt in np.unique(all_tones):
                is_same_tone = np.array([tt == t for t in toneType])
                ok_block = np.any(tone_in_block[np.logical_not(is_same_tone)], axis=0) * np.all(
                    np.logical_not(tone_in_block[is_same_tone]), axis=0)
                try:
                    id_ok = np.random.choice(np.where(ok_block)[0], 100, replace=False)
                except:
                    print("attention, lesss than 100 possible negatives, negatives might be too similar between "
                          "each other to give comparable loss values")
                    try:
                        id_ok = np.random.choice(np.where(ok_block)[0], 100, replace=True)
                    except:
                        raise Exception("")
                negative_dic[tt] = id_ok

            if oneEvalPerEvent:
                negative_masks = []
                for toneblock, tt in zip(tone_in_block, toneType):
                    negative_mask = np.zeros((latent_length, num_negative_samples), dtype=int)
                    for i in np.where(toneblock)[0]:
                        negative_mask[i, :] = negative_dic[tt]
                    negative_masks += [negative_mask]
                negative_mask = np.stack(negative_masks, axis=0)
            else:
                negative_mask = np.zeros((latent_length, num_negative_samples), dtype=int)
                for toneblock, tt in zip(tone_in_block, toneType):
                    negative_mask[toneblock, :] = negative_dic[tt]

            # TODO
            mask_time_indices[id_sound, :, :] = tone_in_block
            sampled_negative_indices[id_sound, :, :] = negative_mask
            id_sound += 1










