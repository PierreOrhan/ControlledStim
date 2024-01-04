import numpy as np
from pathlib import Path

from sounds.perExperiment.protocols.AlRoumi2023 import RandRegRand_LOT,RandRegRand_LOT_Generalize
from sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial

output_dir = Path("/media/pierre/NeuroData2/datasets/lot_testGeneration") / "test_randregrand"

# r1= RandRegRand_LOT(lot_seq="pairs",tones_fs=np.logspace(np.log(50),np.log(400),num=50),
#                    motif_repeat=10,isi=0,sequence_isi=0.2)
rs = []
for isi in np.arange(0,0.3,step=0.05):
    rs += [RandRegRand_LOT(name="RandReg_lot_isi-"+str(isi),lot_seq="pairsAndAlt2",tones_fs=np.logspace(np.log(50),np.log(400),num=50),
                       motif_repeat=10,isi=isi,sequence_isi=0.2)]
lp = ListProtocol_independentTrial(rs)
lp.generate(n_trial=1,output_dir=output_dir)

from sounds.experimentsClass.element_masking import mask_and_latent
mask_and_latent(str(output_dir))
# r.generate(n_trial=10,output_dir=output_dir)