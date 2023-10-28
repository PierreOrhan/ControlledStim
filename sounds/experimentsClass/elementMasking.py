import zarr as zr
import os
from sounds.utils import get_input_lengths
from pandas import Interval
import numpy as np
import tqdm
class ElementMasking():
    @classmethod
    def _mask_and_latent(cls,dirZarr,fileName,duration_tone,oneEvalPerEvent=True,duration_silence = 0):
        ## The tone structure is assumed to be the same, but the negative indices
        # are sampled acorrding to the tone sequence.

        wav2vec2_receptiveField = 400  # in number of input sample that are taken into account in a latent sample
        wav2vec2_stride = 320  # Stride between each latent sample
        wav2vec2_params = {"conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                           "conv_stride": [5, 2, 2, 2, 2, 2, 2]}
        num_negatives = 100

        zg = zr.open_group(os.path.join(dirZarr, fileName, "sounds.zarr"), mode="r")
        sound_mat = zg["sound_mat"]
        tones_seq = zg["tones_sequences"]

        latent_length = get_input_lengths(sound_mat.shape[-1], wav2vec2_params["conv_kernel"],
                                          wav2vec2_params["conv_stride"]).item()

        toneStart = np.array(((duration_tone+duration_silence) / 1000) * np.arange(tones_seq.shape[-1]), dtype=float)
        toneEnd = np.array(((duration_tone+duration_silence) / 1000) * (np.arange(tones_seq.shape[-1]) + 1) - (duration_silence/1000), dtype=float)

        ### AFTER HERE: added.
        latentblock_start = np.arange(0, wav2vec2_stride * latent_length, step=wav2vec2_stride)
        latentblock_end = latentblock_start + wav2vec2_receptiveField
        latentblock_itv = [Interval(s, e, closed="left") for s, e in zip(latentblock_start, latentblock_end)]
        toneStart_sample = np.array(toneStart * 16000, dtype=int)
        toneEnd_sample = np.array(toneEnd * 16000, dtype=int)
        tone_itv = [Interval(s, e, closed="left") for s, e in zip(toneStart_sample, toneEnd_sample)]
        # Find the block inside with overlap with the tones
        tone_in_block = np.array([[ti.overlaps(lti) for lti in latentblock_itv] for ti in tone_itv])

        block_inside_tone = np.array(
            [[ti.left <= lti.left and ti.right >= lti.right for lti in latentblock_itv] for ti in tone_itv])
        assert np.all(np.any(block_inside_tone, axis=-1))
        # A block that is fully contained in the tone
        ltrs = np.stack([block_inside_tone for _ in range(tones_seq.shape[0])], axis=0)
        ## stack: we have the same structure for all the sounds here (one type of sequence), so the
        # focus of the loss will be on the same latent.

        zg = zr.open_group(os.path.join(dirZarr, fileName, "sounds.zarr"), mode="a")
        if "mask_time_indices" not in zg.keys():
            zg.create("mask_time_indices", shape=(tones_seq.shape[0], tones_seq.shape[1], latent_length),
                      chunks=(1, None, None), dtype=bool)
            if oneEvalPerEvent:
                zg.create("sampled_negative_indices",
                          shape=(tones_seq.shape[0],tones_seq.shape[1], latent_length, num_negatives),
                          chunks=(1,None,None, None), dtype=int)
            else:
                zg.create("sampled_negative_indices",
                          shape=(tones_seq.shape[0], latent_length, num_negatives),
                          chunks=(1, None, None), dtype=int)
            # zg.array("latent_time_reduction", data=(tones_seq.shape[0],tones_seq.shape[1],latent_length),
            #          chunks=(1, None, None),dtype=bool)
            zg.array("latent_time_reduction", data=ltrs,chunks=(1, None, None))

        for id_sound, toneType in tqdm.tqdm(enumerate(tones_seq)):
            negative_dic = {}
            for tt in np.unique(tones_seq):
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
                    negative_mask = np.zeros((latent_length, num_negatives), dtype=int)
                    for i in np.where(toneblock)[0]:
                        negative_mask[i, :] = negative_dic[tt]
                    negative_masks += [negative_mask]
                negative_mask = np.stack(negative_masks,axis=0)
            else:
                negative_mask = np.zeros((latent_length, num_negatives), dtype=int)
                for toneblock, tt in zip(tone_in_block, toneType):
                    negative_mask[toneblock, :] = negative_dic[tt]

            zg["mask_time_indices"][id_sound, ...] = tone_in_block
            zg["sampled_negative_indices"][id_sound, ...] = negative_mask


    @classmethod
    def _mask_and_latent_PerSound(cls,dirZarr,fileName,oneEvalPerEvent=True):

        zgEvents = zr.open_group(os.path.join(dirZarr,fileName, "events.zarr"), mode="r")
        zg = zr.open_group(os.path.join(dirZarr, fileName, "sounds.zarr"), mode="a")
        sound_mat = zg["sound_mat"]

        ######### Definition of the masks
        wav2vec2_receptiveField = 400  # in number of input sample that are taken into account in a latent sample
        wav2vec2_stride = 320  # Stride between each latent sample
        wav2vec2_params = {"conv_kernel": [10, 3, 3, 3, 3, 2, 2],
                           "conv_stride": [5, 2, 2, 2, 2, 2, 2]}
        num_negatives = 100

        mask_time_indices = []
        negative_mask_indices = []
        ltrs = []
        for sound_id in tqdm.tqdm(range(zgEvents["toneStart"].shape[0])):
            toneStart = np.array([float(t[-1]) for t in zgEvents["toneStart"][sound_id, :]])
            toneEnd = np.array([float(t[-1]) for t in zgEvents["toneEnd"][sound_id, :]])
            toneType = np.array([t[1] for t in zgEvents["toneStart"][sound_id, :]])

            # Wav2vec2 downsamples to 49 Hz the waveform signal and the masks are defined in this latent space:
            latent_length = get_input_lengths(sound_mat.shape[-1], wav2vec2_params["conv_kernel"],
                                              wav2vec2_params["conv_stride"]).item()

            # For each latent we compute the start and end of its receptive field:
            latentblock_start = np.arange(0, wav2vec2_stride * latent_length, step=wav2vec2_stride)
            latentblock_end = latentblock_start + wav2vec2_receptiveField
            latentblock_itv = [Interval(s, e, closed="left") for s, e in zip(latentblock_start, latentblock_end)]

            toneStart_sample = np.array(toneStart * 16000, dtype=int)
            toneEnd_sample = np.array(toneEnd * 16000, dtype=int)
            tone_itv = [Interval(s, e, closed="left") for s, e in zip(toneStart_sample, toneEnd_sample)]

            # Find the block inside with overlap with the tones
            tone_in_block = np.array([[ti.overlaps(lti) for lti in latentblock_itv] for ti in tone_itv])

            block_inside_tone = np.array(
                [[ti.left <= lti.left and ti.right >= lti.right for lti in latentblock_itv] for ti in tone_itv])
            assert np.all(np.any(block_inside_tone, axis=-1))
            # A block that is fully contained in the tone
            ltrs += [block_inside_tone]

            negative_dic = {}
            for tt in np.unique(toneType):
                is_same_tone = np.array([tt == t for t in toneType])
                ok_block = np.any(tone_in_block[np.logical_not(is_same_tone)], axis=0) * np.all(
                    np.logical_not(tone_in_block[is_same_tone]), axis=0)
                try:
                    id_ok = np.random.choice(np.where(ok_block)[0], 100, replace=False)
                except:
                    print("attention, lesss than 100 possible negatives, negatives might be too similar between "
                          "each other to give comparable loss values")
                    id_ok = np.random.choice(np.where(ok_block)[0], 100, replace=True)
                negative_dic[tt] = id_ok

            if oneEvalPerEvent:
                negative_masks = []
                for toneblock, tt in zip(tone_in_block, toneType):
                    negative_mask = np.zeros((latent_length, num_negatives), dtype=int)
                    new_mask = np.zeros(latent_length, dtype=bool)
                    new_mask[toneblock] = True
                    masks += [new_mask]
                    for i in np.where(toneblock)[0]:
                        negative_mask[i, :] = negative_dic[tt]
                    negative_masks += [negative_mask]
                negative_mask = np.stack(negative_masks,axis=0)
            else:
                masks = []
                negative_mask = np.zeros((latent_length, num_negatives), dtype=int)
                for toneblock, tt in zip(tone_in_block, toneType):
                    new_mask = np.zeros(latent_length, dtype=bool)
                    new_mask[toneblock] = True
                    masks += [new_mask]
                    negative_mask[toneblock, :] = negative_dic[tt]

            mask_time_indices += [np.stack(masks, axis=0)]
            negative_mask_indices += [negative_mask]
        mask_time_indices = np.stack(mask_time_indices)
        negative_mask_indices = np.stack(negative_mask_indices)
        ltrs = np.stack(ltrs)

        zg = zr.open_group(os.path.join(dirZarr, "test", "sounds.zarr"), mode="a")
        if "mask_time_indices" not in zg.keys():
            zg.array("mask_time_indices",
                     data=mask_time_indices,
                     chunks=(None, None, None))
        else:
            zg["mask_time_indices"] = mask_time_indices
        if "sampled_negative_indices" not in zg.keys():
            zg.array("sampled_negative_indices", data=negative_mask_indices, chunks=(1, None, None))
        else:
            zg["sampled_negative_indices"] = negative_mask_indices
        if "latent_time_reduction" not in zg.keys():
            zg.array("latent_time_reduction", data=ltrs, chunks=(None, None, None))
        else:
            zg["latent_time_reduction"] = ltrs