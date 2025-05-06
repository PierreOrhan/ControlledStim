from pathlib import Path
from sounds.perExperiment.protocols.Bekinschtein2009 import RandRegDev_LocalGlobal,RandRegDev_LocalGlobal_orig
from sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial
from sounds.perExperiment.sequences.lot_patterns import lot_patterns
### Debugging: We fix the pool and repeat the probing over and over
import pandas as pd
from sounds.perExperiment.sound_elements import Bip,Silence
from sounds.perExperiment.sound_elements import Sound_pool,Sound
import numpy as np


motif_repeat = 10
output_dir = Path("/media/pierre/NeuroData2/datasets/lot_further/localglobal") / "localglobal_extremeFreq"

tones_fs=np.logspace(np.log(222),np.log(2000),20,base=np.exp(1))
sounds_rand = [Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f]) for idf, f in
          enumerate(np.array(tones_fs[:16])[[3, 11, 14,  2,  7,  9,  1, 13,  8,  5,  6, 12,  4, 10, 15]])]
s_rand = Sound_pool.from_list(sounds_rand)

low_freq = 222
high_freq = 2000
isi = 0.1
seq_isi = 0.5

rs = []
for k in ["localstandard","localdeviant"]:
    s_reg = Sound_pool.from_list([Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f])
                                  for idf, f in enumerate([low_freq, high_freq])])

    rs += [RandRegDev_LocalGlobal(name=k+"_globalDeviant"+"_highlow",
                               global_standard=k,
                               tones_fs=tones_fs,
                               motif_repeat=motif_repeat,
                               isi=isi,sequence_isi=seq_isi)]
    rs[-1].fixPoolSampled(s_rand, s_reg)

    rs += [RandRegDev_LocalGlobal_orig(name=k+"_globalStandard"+"_highlow",
                                global_standard=k,
                                tones_fs=tones_fs,
                                motif_repeat=motif_repeat,
                               isi=isi,sequence_isi=seq_isi)]
    rs[-1].fixPoolSampled(s_rand, s_reg)

    s_reg = Sound_pool.from_list([Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f])
                                  for idf,f in  enumerate([high_freq,low_freq])])

    rs += [RandRegDev_LocalGlobal(name=k+"_globalDeviant"+"_lowhigh",
                               global_standard=k,
                               tones_fs=tones_fs,
                               motif_repeat=motif_repeat,
                               isi=isi,sequence_isi=seq_isi)]
    rs[-1].fixPoolSampled(s_rand, s_reg)

    rs += [RandRegDev_LocalGlobal_orig(name=k+"_globalStandard"+"_lowhigh",
                               global_standard=k,
                               tones_fs=tones_fs,
                               motif_repeat=motif_repeat,
                               isi=isi,sequence_isi=seq_isi)]
    rs[-1].fixPoolSampled(s_rand, s_reg)

lp = ListProtocol_independentTrial(rs)
lp.generate(n_trial=1,output_dir=output_dir)
from sounds.experimentsClass.element_masking import mask_and_latent_BalancedNegatives
mask_and_latent_BalancedNegatives(str(output_dir))
