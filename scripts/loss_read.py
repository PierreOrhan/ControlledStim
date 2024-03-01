import shutil

import zarr
import numpy as np
import matplotlib.pyplot as plt

from sounds.perExperiment.protocols.Benjamin2023 import Benjamin2023
from sounds.perExperiment.sequences.patterns import FullCommunityGraph, SparseCommunityGraph, HighSparseCommunityGraph
from probe.scripts.postModule import main_Emile as postModule
from sounds.experimentsClass.element_masking import mask_and_latent

walk_length = 200
isi = 0.05
data_path = "/Users/Emile/PycharmProjects/ControlledStim/sounds/data/"
save_dir = "/Users/Emile/PycharmProjects/ControlledStim/sounds/data/LossData/"

full_protocol = Benjamin2023()
full_protocol.seq = FullCommunityGraph(walk_length=walk_length, isi = isi)
full_protocol.walk_length = walk_length
full_protocol.name = "FullCommunity"
full_protocol.generate(1,data_path + full_protocol.name)
mask_and_latent(data_path + full_protocol.name)
postModule(save_dir = save_dir + full_protocol.name,data_dir = data_path + full_protocol.name)
shutil.rmtree(save_dir + full_protocol.name + "/loss_array/", ignore_errors=True)
shutil.copytree(save_dir + full_protocol.name + "/onlylast/postAnalyses_loss.zarr" + data_path + full_protocol.name,save_dir + full_protocol.name + "/loss_array/")
shutil.rmtree(save_dir + full_protocol.name + "/onlylast/")


sparse_protocol = Benjamin2023()
sparse_protocol.seq = SparseCommunityGraph(walk_length=walk_length, isi = isi)
sparse_protocol.name = "SparseCommunity"
sparse_protocol.walk_length = walk_length
sparse_protocol.generate(1,data_path + sparse_protocol.name)
mask_and_latent(data_path + sparse_protocol.name)
postModule(save_dir = save_dir + sparse_protocol.name,data_dir = data_path + sparse_protocol.name)
shutil.rmtree(save_dir + sparse_protocol.name + "/loss_array/", ignore_errors=True)
shutil.copytree(save_dir + sparse_protocol.name + "/onlylast/postAnalyses_loss.zarr" + data_path + sparse_protocol.name,save_dir + sparse_protocol.name + "/loss_array/")
shutil.rmtree(save_dir + sparse_protocol.name + "/onlylast/")

high_sparse_protocol = Benjamin2023()
high_sparse_protocol.seq = HighSparseCommunityGraph(walk_length=walk_length, isi = isi)
high_sparse_protocol.name = "HighSparseCommunity"
high_sparse_protocol.walk_length = walk_length
high_sparse_protocol.generate(1,data_path + high_sparse_protocol.name)
mask_and_latent(data_path + high_sparse_protocol.name)
postModule(save_dir = save_dir + high_sparse_protocol.name,data_dir = data_path + high_sparse_protocol.name)
shutil.rmtree(save_dir + high_sparse_protocol.name + "/loss_array/", ignore_errors=True)
shutil.copytree(save_dir + high_sparse_protocol.name + "/onlylast/postAnalyses_loss.zarr" + data_path + high_sparse_protocol.name,save_dir + high_sparse_protocol.name + "/loss_array/")
shutil.rmtree(save_dir + high_sparse_protocol.name + "/onlylast/")


t_full = zarr.open(save_dir + full_protocol.name + "/loss_array")
t_sparse = zarr.open(save_dir + sparse_protocol.name + "/loss_array")
t_high_sparse = zarr.open(save_dir + high_sparse_protocol.name + "/loss_array")

loss_full = np.squeeze(t_full)
loss_sparse = np.squeeze(t_sparse)
loss_high_sparse = np.squeeze(t_high_sparse)

# mean_loss = np.mean(loss, axis=0)
# plt.plot(mean_loss)
# plt.title("Mean loss over 10 trials")
# plt.show()
plt.plot(loss_full)
plt.plot(loss_sparse)
plt.plot(loss_high_sparse)
plt.legend(["Full", "Sparse", "High Sparse"])
plt.title("Losses for walk_length = %dwalk_length, isi = %isi" % (walk_length, isi))
plt.show()


