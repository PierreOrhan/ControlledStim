from torch.utils.data import Dataset
from datasets import IterableDataset
from pathlib import Path
import pandas as pd
import soundfile as sf
import zarr as zr
import numpy as np

def load_ANNdataset_withMask(dataset_dir : Path):
    sequences = pd.read_csv(dataset_dir / "sequences.csv")
    def gen(shards):
        for shard in shards:
            sequence = sequences.iloc[shard, :]
            sd,sr = sf.read(sequence["wav_path"])
            zg = zr.open_group(sequence["mask_info_path"])
            for element_id in range(zg["mask_time_indices"].shape[0]):
                datapoint = {"input_values":sd,
                             "mask_time_indices":zg["mask_time_indices"][element_id,:],
                             "sampled_negative_indices":zg["sampled_negative_indices"][element_id,:],
                             "latent_attention_mask" : zg["latent_attention_mask"][element_id,:]}
                yield datapoint

    shards = np.arange(sequences.shape[0])
    ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})
    ds = ds.with_format("torch")
    # ---> Transforms to a TorchIterableDataset which has the attribute len and can be used
    # in a DataLoader (i.e combined with a collator!!)
    ds.info.dataset_size = np.sum(sequences["number_element"])