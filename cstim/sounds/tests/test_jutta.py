from pathlib import Path
from sounds.perExperiment.protocols.Jutta2012 import PitchRuleDeviant_1
from sounds.experimentsClass.element_masking import mask_and_latent_BalancedNegatives

output_dir = Path("/media/pierre/Crucial X8/datasets/BoubenecLab/Jutta2012") / "Jutta2012_small"
protocol = PitchRuleDeviant_1(samplerate = 16000,nb_deviant_pitch=2,nb_standard=2,nb_deviant_rule=2)
protocol.generate(n_trial=10,output_dir=output_dir)
mask_and_latent_BalancedNegatives(str(output_dir))


output_dir = Path("/media/pierre/Crucial X8/datasets/BoubenecLab/Jutta2012") / "Jutta2012_large"
protocol = PitchRuleDeviant_1(samplerate = 16000)
protocol.generate(n_trial=10,output_dir=output_dir)
mask_and_latent_BalancedNegatives(str(output_dir))