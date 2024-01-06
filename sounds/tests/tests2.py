import numpy as np
from pathlib import Path

from sounds.perExperiment.protocols.Barascud2016 import RandRegRand
from sounds.perExperiment.protocols.Benjamin2023 import Benjamin2023_syllable

output_dir = Path("C:\\Users\orhan\Documents\data\datasets") / "test_gen2"
# r = RandRegRand(sequence_isi=0,motif_repeat=5,
#                 tones_fs=np.logspace(np.log(50),np.log(400),num=50))
# r.generate(n_trial=1,output_dir=output_dir)

r= Benjamin2023_syllable()
r.generate(n_trial=1,output_dir=output_dir)