import numpy as np
import pandas as pd
from datasets import Dataset

# Creating huggingface dataset with sounds
# There should be 3 different features: mask_time_indices, sampled_negative_indices, latent_time_reduction
# respectively of dimension 2,3,2

# mask_time_indices: (number_of_masks,latent_length)
# sampled_negative_indices: (number_of_masks,latent_length,number_of_negative_samples)
# latent_time_reduction: (number_of_masks,latent_length)

mask_time_indices = np.zeros((10, 100))
sampled_negative_indices = np.zeros((10, 100, 3))
latent_time_reduction = np.zeros((10, 100))

dataset = {
    "mask_time_indices": mask_time_indices,
    "sampled_negative_indices": sampled_negative_indices,
    "latent_time_reduction": latent_time_reduction
}

# Creating huggingface dataset with sounds
dataset = Dataset.from_dict(dataset)

print(dataset)
