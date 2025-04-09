from pathlib import Path
from cstim.sounds.perExperiment.protocols.Saffran1996 import Saffran_otherStim
from cstim.sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial

### Debugging: We fix the pool and repeat the probing over and over
import os
# output_dir = Path("/media/pierre/NeuroData2/datasets/lot_further/lot_decoding_unitnormV2") / ("randregrand"+str(motif_repeat)+"_noIsi")
output_dir = Path("/auto/data5/speechExposureEphys/LOT/lot_further/lot_decoding_unitnormVCompBio")/("SaffranOtherStim_debug")
os.makedirs(Path("/auto/data5/speechExposureEphys/LOT/lot_further/lot_decoding_unitnormVCompBio"),exist_ok=True)


# sound_paths = {"speech": ,
#                 "music": ,
#                 "env": ,}

lp = ListProtocol_independentTrial([Saffran_otherStim()])
lp.generate(n_trial=10,output_dir=output_dir)
