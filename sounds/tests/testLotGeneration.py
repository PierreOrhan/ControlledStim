import numpy as np
from pathlib import Path

from sounds.perExperiment.protocols.AlRoumi2023 import RandRegRand_LOT

output_dir = Path("/media/pierre/NeuroData2/datasets/lot_testGeneration") / "test_randregrand"

r= RandRegRand_LOT(lot_seq="pairs",tones_fs=np.logspace(np.log(50),np.log(400),num=50),
                   motif_repeat=10)
r.generate(n_trial=10,output_dir=output_dir)