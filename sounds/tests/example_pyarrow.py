import pyarrow as pa
import numpy as np
import datasets

datasets.Sequence(datasets.Value("float32"))


b= [np.zeros(20,30)]

mask_time_indices = pa.Array(np.zeros(20, 256), dtype=bool)
sampled_negative_indices = pa.Array(np.zeros(20,256, 100),dtype=int)
latent_time_reduction = pa.Array(np.zeros(20,256),dtype=int)

tb = pa.Table([mask_time_indices,sampled_negative_indices,latent_time_reduction],
         name=["mask_time_indices","sampled_negative_indices","latent_time_reduction"])