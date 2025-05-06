from pathlib import Path
from cstim.sounds.perExperiment.protocols.Saffran1996 import Saffran,Saffran_StressClue
from cstim.sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial

### Debugging: We fix the pool and repeat the probing over and over
import os

# output_dir = Path("/media/pierre/NeuroData2/datasets/lot_further/lot_decoding_unitnormV2") / ("randregrand"+str(motif_repeat)+"_noIsi")
output_dir = Path("/auto/data5/speechExposureEphys/LOT/lot_further/lot_decoding_unitnormVCompBio")/("SaffranStress_debug")
os.makedirs(Path("/auto/data5/speechExposureEphys/LOT/lot_further/lot_decoding_unitnormVCompBio"),exist_ok=True)

lp = ListProtocol_independentTrial([Saffran_StressClue()])
lp.generate(n_trial=10,output_dir=output_dir)

from cstim.sounds.experimentsClass.element_maskingV2 import mask_latentV2,mask_and_latent_BalancedNegativesV2
mask_and_latent_BalancedNegativesV2(str(output_dir))