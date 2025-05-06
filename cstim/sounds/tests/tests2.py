import numpy as np
from pathlib import Path
import os
from sounds.perExperiment.protocols.Barascud2016 import RandRegRand
from sounds.perExperiment.protocols.Benjamin2023 import Benjamin2023_syllable
from sounds.perExperiment.protocols.Jutta2012 import PitchRuleDeviant_1

#output_dir = Path("C:/Users/juliengadonneix/Desktop/code/LongProject/Jutta2012") / "test_gen2"
output_dir = Path("/home/pierre/Documents/genJulien2")
os.makedirs(output_dir,exist_ok=True)
os.makedirs(output_dir / "sounds",exist_ok=True)
os.makedirs(output_dir / "sound_info",exist_ok=True)

# r = RandRegRand(sequence_isi=0,motif_repeat=5,
#                 tones_fs=np.logspace(np.log(50),np.log(400),num=50))
# r.generate(n_trial=1,output_dir=output_dir)

# r= Benjamin2023_syllable()
# r.generate(n_trial=1,output_dir=output_dir)

r = PitchRuleDeviant_1()
r.generate(n_trial=20,output_dir=output_dir)

from sounds.experimentsClass.element_masking import mask_and_latent_BalancedNegatives
mask_and_latent_BalancedNegatives(str(output_dir))
