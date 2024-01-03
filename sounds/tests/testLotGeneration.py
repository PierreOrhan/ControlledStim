import numpy as np
from pathlib import Path

from sounds.perExperiment.protocols.AlRoumi2023 import RandRegRand_LOT,RandRegRand_LOT_Generalize
from sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial

output_dir = Path("/media/pierre/NeuroData2/datasets/lot_testGeneration") / "test_randregrand"

r1= RandRegRand_LOT(lot_seq="pairs",tones_fs=np.logspace(np.log(50),np.log(400),num=50),
                   motif_repeat=10)
r2= RandRegRand_LOT(lot_seq="centermirror",tones_fs=np.logspace(np.log(50),np.log(400),num=50),
                   motif_repeat=10)
lp = ListProtocol_independentTrial([r1,r2])
lp.generate(n_trial=2,output_dir=output_dir)

from sounds.experimentsClass.element_masking import mask_and_latent
mask_and_latent(str(output_dir))
# r.generate(n_trial=10,output_dir=output_dir)