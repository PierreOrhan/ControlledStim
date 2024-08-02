from pathlib import Path
from sounds.perExperiment.protocols.AlRoumi2023 import LOT,LOT_generalize,LOT_deviant
from sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial
from sounds.perExperiment.sequences.lot_patterns import lot_patterns
import numpy as np
from sounds.experimentsClass.element_masking import mask_latent

tones_fs=np.logspace(np.log(222),np.log(2000),20,base=np.exp(1))
output_dir = Path("/media/pierre/NeuroData2/datasets/lot_further/lot_treedist_fast")
# rs = []
# for k in lot_patterns.keys():
#     rs+=[LOT(name=k+"_lot",
#                lot_seq=k,
#                tones_fs=tones_fs,
#                motif_repeat=10,
#                isi=0.0,sequence_isi=0.0)]
# lp = ListProtocol_independentTrial(rs)
# lp.generate(n_trial=20,output_dir=output_dir /"habituation")
# mask_latent(str(output_dir  /"habituation"))
# #
# rs = []
# for k in lot_patterns.keys():
#     rs+=[LOT_generalize(name=k+"_lot",
#                lot_seq=k,
#                tones_fs=tones_fs,
#                motif_repeat=10,
#                isi=0.0,sequence_isi=0.0)]
# lp = ListProtocol_independentTrial(rs)
# lp.generate(n_trial=20,output_dir=output_dir /"habituationgeneralize")
# mask_latent(str(output_dir /"habituationgeneralize"))
rs =[]
for idf,(f0,f1) in enumerate(zip([350,390,415,440],[500,550,580,622])):
    tones_fs = np.array([[f0,f0*2,f0*3],[f1,f1*2,f1*3]])
    rs += [LOT_deviant(name=str(idf)+"lot",
                tones_fs=tones_fs,
                motif_repeat=10,
                isi=0.0, sequence_isi=0.0)]
lp = ListProtocol_independentTrial(rs)
lp.generate(n_trial=1*10*10,output_dir=output_dir /"habituationWithTest") # 10 different sequences
mask_latent(str(output_dir  /"habituationWithTest"))