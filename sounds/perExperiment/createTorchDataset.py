from torch.utils.data import Dataset
from datasets import IterableDataset
from pathlib import Path
import pandas as pd
import soundfile as sf
import zarr as zr
import numpy as np

def load_ANNdataset_withMask(dataset_dir : Path,partially_causal = True,extendWithMask=True) -> IterableDataset:
    """
    Converts an ANNdataset to a torch dataset, usable in pytorch.
    :param dataset_dir: directory toward a dataset as formatted by ControlledStim package.
    :param partially_causal: adds a latent_attention_mask (vector) to the model inputs, which can be used to generate
    a mask at the level of attention to prevent the attention on future token, be careful to also use it to set to 0 any
    positional encoding that would use local convolution.
    :param extendWithMask: if we yield the sound for every different masks associated to the sounds
    :return:
        Huggingface's datasets IterableDataset, with dataset.info.dataset_size filed to allow it to be used by DataLoader.
    """
    sequences = pd.read_csv(dataset_dir / "trials.csv")
    def gen(shards):
        for shard in shards:
            sequence = sequences.iloc[shard, :]
            sd,sr = sf.read(sequence["wav_path"])
            zg = zr.open_group(sequence["mask_info_path"],mode="r")
            if extendWithMask:
                mti = zg["mask_time_indices"][shard, ...]
                sni = zg["sampled_negative_indices"][shard, ...]
                if "latent_time_reduction" in zg.keys():
                    ltr = zg["latent_time_reduction"][shard, ...]
                    has_ltr = True
                else:
                    has_ltr = False
                for element_id in range(zg["mask_time_indices"].shape[0]):
                    if partially_causal:
                        ## Adds attention masking, to forbid the use of future elements:
                        attention_mask = np.zeros(mti.shape[-1],dtype=bool) + True
                        end_mask = np.where(mti[element_id,...])[0][-1]
                        attention_mask[end_mask+1:] = False

                        if has_ltr:
                            yield {"input_values": sd, "mask_time_indices": mti[element_id, ...],
                                   "sampled_negative_indices": sni[element_id, ...],
                                   "latent_attention_mask":attention_mask,
                                   "latent_time_reduction":ltr[element_id,...]}
                        else:
                            yield {"input_values": sd, "mask_time_indices": mti[element_id, ...],
                                   "sampled_negative_indices": sni[element_id, ...],
                                   "latent_attention_mask":attention_mask}
                    else:
                        if has_ltr:
                            yield {"input_values": sd, "mask_time_indices": mti[element_id, ...],
                                   "sampled_negative_indices": sni[element_id, ...],
                                   "latent_time_reduction":ltr[element_id,...]}
                        else:
                            yield {"input_values": sd, "mask_time_indices": mti[element_id, ...],
                                   "sampled_negative_indices": sni[element_id, ...]}
            else:
                if not partially_causal:
                    raise Exception("causal reading of the activity without providing the mask is not implemented")
                yield {"input_values": sd}

    shards = np.arange(sequences.shape[0])
    ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
    ds = ds.with_format("torch")
    # ---> Transforms to a TorchIterableDataset (named IterableDataset in pytorch)
    # which has the attribute len and can be used
    # in a DataLoader (i.e combined with a collator!!)
    ds.info.dataset_size = np.sum(sequences["number_element"])
    return ds