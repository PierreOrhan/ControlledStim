import pandas as pd
from pathlib import Path
from sounds.utils import get_input_lengths
import soundfile as sf
import numpy as np

def mask_and_latent(sequence_data_set_dir: str, oneEvalPerEvent: bool = True):
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
    sequences = pd.read_csv(sequence_data_set_dir + "/trials.csv")
    for seq in range(sequences.shape[0]):
        sequence = sequences.iloc[seq, :]
        sequence_info = pd.read_csv(sequence["sound_info_path"])

        sound_mat = sf.read(sequence["wav_path"])

        # Get the number of latent samples
        latent_length = get_input_lengths(len(sound_mat[0]), wav2vec2_params["conv_kernel"],
                                          wav2vec2_params["conv_stride"])

        # define the intervals of each tone in temporal space
        tone_start = sequence_info["start"][sequence_info["name"]!="Silence"]
        tone_duration = sequence_info["duration"][sequence_info["name"]!="Silence"]
        tone_end = tone_start + tone_duration

        # define the intervals of each tone in latent space
        latentblock_start = np.arange(0, wav2vec2_stride * latent_length, step=wav2vec2_stride)
        latentblock_end = latentblock_start + wav2vec2_receptiveField
        latentblock_itv = [pd.Interval(s, e, closed="left") for s, e in
                           zip(latentblock_start, latentblock_end)]
        toneStart_sample = np.array(tone_start * 16000, dtype=int)
        toneEnd_sample = np.array(tone_end * 16000, dtype=int)
        tone_itv = [pd.Interval(s, e, closed="left") for s, e in zip(toneStart_sample, toneEnd_sample)]

        # For all tones, find the blocks with which they overlap
        tone_in_block: np.ndarray[bool] = np.array(
            [[ti.overlaps(lti) for lti in latentblock_itv] for ti in tone_itv])

        block_inside_tone = np.array(
            [[ti.left <= lti.left and ti.right >= lti.right for lti in latentblock_itv] for ti in
             tone_itv])

        silence_start = sequence_info["start"][sequence_info["name"]=="Silence"] * 16000
        silence_duration = sequence_info["duration"][sequence_info["name"]=="Silence"] * 16000
        silence_end = silence_start + silence_duration

        silence_itv = [pd.Interval(s, e, closed="left") for s, e in zip(silence_start, silence_end)]
        block_inside_silence: np.ndarray[bool] = np.array(
            [[ti.left <= lti.left and ti.right >= lti.right for lti in latentblock_itv] for ti in silence_itv])

        assert np.all(np.any(block_inside_tone, axis=-1))
        nb_tones = np.sum(sequence_info["name"]!="Silence")

        # Blocks that are fully contained in the tone
        latent_time_reduction_blocks = block_inside_tone #np.stack([block_inside_tone for _ in range(nb_tones)], axis=0)

        ## stack: we have the same structure for all the sounds here (one type of sequence), so the
        # focus of the loss will be on the same latent.
        mask_time_indices = np.zeros((nb_tones, latent_length), dtype=bool)
        sampled_negative_indices = np.zeros((nb_tones, latent_length, num_negative_samples),
                                            dtype=int)

        toneType = sequence_info["name"][sequence_info["name"]!="Silence"].to_numpy()
        negative_dic = {}
        for tt in np.unique(sequence_info["name"]):
            is_same_tone = np.array([tt == t for t in toneType])
            ok_block = np.any(tone_in_block[np.logical_not(is_same_tone)], axis=0) * np.all(
                np.logical_not(tone_in_block[is_same_tone]), axis=0)
            try:
                id_ok = np.random.choice(np.where(ok_block)[0], num_negative_samples, replace=False)
            except:
                print("attention, lesss than 100 possible negatives, negatives might be too similar between "
                      "each other to give comparable loss values")
                try:
                    id_ok = np.random.choice(np.where(ok_block)[0], 100, replace=True)
                except:
                    print("we authorize using silence as negatives")
                    ok_block = np.any(block_inside_silence)
                    id_ok = np.random.choice(np.where(ok_block)[0], num_negative_samples, replace=True)
            negative_dic[tt] = id_ok

        negative_masks = []
        for toneblock, tt in zip(tone_in_block, toneType):
            negative_mask = np.zeros((latent_length, num_negative_samples), dtype=int)
            for i in np.where(toneblock)[0]:
                negative_mask[i, :] = negative_dic[tt]
            negative_masks += [negative_mask]

        mat_negative_mask = np.stack(negative_masks, axis=0)

        mask_time_indices[:, :] = tone_in_block
        sampled_negative_indices[:, :, :] = mat_negative_mask

        import zarr as zr
        zg = zr.open_group(Path(sequence_data_set_dir) / sequence["wav_path"].replace(".wav",".masks"), mode="w")
        # if "mask_time_indices" not in zg.keys():
        zg.array("mask_time_indices",data=mask_time_indices,chunks=(None,None))
        zg.array("sampled_negative_indices", data=sampled_negative_indices, chunks=(None,None,None))
        zg.array("latent_time_reduction", data=latent_time_reduction_blocks, chunks=(None,None))

        ## We update the masks info_path in the dataframe
        sequences.loc[seq,"mask_info_path"] = str(Path(sequence_data_set_dir) / sequence["wav_path"].replace(".wav",".masks"))
    # update the dataset info csv:
    sequences.to_csv(sequence_data_set_dir + "/trials.csv")