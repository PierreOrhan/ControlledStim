import numpy as np
from pathlib import Path

from sounds.perExperiment.protocols.AlRoumi2023 import RandRegRand_LOT,RandRegRand_LOT_Generalize,RandRegRand_LOT_deviant,\
    RandRegRand_LOT_deviant_BoundFixedPool,RandRegRand_LOT_deviant_fixedPool
from sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial

output_dir = Path("/media/pierre/NeuroData2/datasets/lot_testGeneration") / "test_randregrand"

# r1= RandRegRand_LOT(lot_seq="pairs",tones_fs=np.logspace(np.log(50),np.log(400),num=50),
#                    motif_repeat=10,isi=0,sequence_isi=0.2)
# for seq_isi in np.arange(0,0.2,step=0.05):
#     for isi in np.arange(0,0.3,step=0.05):
# seq_isi = 0.5
# isi = 0.2
seq_isi = 0
isi = 0.02
rs = [RandRegRand_LOT_deviant_BoundFixedPool(name="RandReg_lot_isi-"+str(isi)+"_seqisi-"+str(seq_isi)+"_deviant-"+str(0),
                           lot_seq="pairsAndAlt2",
                           tones_fs=np.logspace(np.log(1000)/np.log(10),np.log(2000)/np.log(10),num=1000),
                           deviant=0,
                           motif_repeat=10,
                           isi=isi,sequence_isi=seq_isi)]
rs[0].samplePool(200,np.inf)
### Make a plot as a function of the distance between the two tones:
for deviant in np.arange(1,4):
    r = RandRegRand_LOT_deviant_BoundFixedPool(name="RandReg_lot_isi-"+str(isi)+"_seqisi-"+str(seq_isi)+"_deviant-"+str(deviant),
                           lot_seq="pairsAndAlt2",
                           tones_fs=np.logspace(np.log(1000)/np.log(10),np.log(2000)/np.log(10),num=1000),
                           deviant=deviant,
                           motif_repeat=10,
                           isi=isi,sequence_isi=seq_isi)
    r.fixPoolSampled(rs[0].s_rand,rs[0].s_reg)
    rs += [r]

lp = ListProtocol_independentTrial(rs)
lp.generate(n_trial=1,output_dir=output_dir)

from sounds.experimentsClass.element_masking import mask_and_latent
mask_and_latent(str(output_dir))
