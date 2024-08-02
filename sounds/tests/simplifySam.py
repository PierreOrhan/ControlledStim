import os

# from sounds.experimentsClass.converting import simplify_subfolders
# from pathlib import Path
# initial_dir = Path("/media/pierre/NeuroData2/datasets/SamLaboratory/wavs")
# for e in os.listdir(initial_dir):
#     simplify_subfolders(initial_dir/e)

from sounds.experimentsClass.converting import fromDir_toDataset
from pathlib import Path
initial_dir = Path("/media/pierre/NeuroData2/datasets/SamLaboratory/wavs")
output_dir = Path("/media/pierre/NeuroData2/datasets/SamLaboratory/wavsOut")
for e in os.listdir(initial_dir):
    fromDir_toDataset(initial_dir/e,output_dir/e,inplace=False)