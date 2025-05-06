from pathlib import Path

from sounds.perExperiment.protocols.AlRoumi2023 import RandRegRand_LOT,RandRegRand_LOT_deviant,RandRegRand_LOT_orig
from sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial
from sounds.perExperiment.sequences.lot_patterns import lot_patterns
### Debugging: We fix the pool and repeat the probing over and over
import pandas as pd
from sounds.perExperiment.sound_elements import Bip,Silence
from sounds.perExperiment.sound_elements import Sound_pool,Sound
import numpy as np


tones_fs=np.logspace(np.log(222),np.log(2000),20,base=np.exp(1))
sounds_rand = [Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f]) for idf, f in
          enumerate(np.array(tones_fs[:16])[ [3, 11, 14,  2,  7,  9,  1, 13,  8,  5,  6,  0, 12,  4, 10, 15]])]
s_rand = Sound_pool.from_list(sounds_rand)

for seq_isi in [0, 0.5]:
    output_dir = Path("/media/pierre/NeuroData2/datasets/lot_further/lot_repeat") / str(seq_isi)
    isi = 0
    rs = []
    for idlow,t_low in enumerate(tones_fs[:-1]):

            for deviant in [1]:
                s_reg = Sound_pool.from_list([Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f])
                                              for idf,f in  enumerate([t_low,2000])]) #889.8296701050009, 1413.4860237345383
                rs += [RandRegRand_LOT_deviant(name="repeat_lot_Deviant_isi-"+str(isi)+"_seqisi-"+str(seq_isi)+"_freq-"+str(idlow),
                                           lot_seq="repeat",
                                           tones_fs=tones_fs,
                                           deviant=deviant,
                                           motif_repeat=10,
                                           isi=isi,sequence_isi=seq_isi)]
                rs[-1].fixPoolSampled(s_rand, s_reg)

    lp = ListProtocol_independentTrial(rs)
    lp.generate(n_trial=1,output_dir=output_dir)
    from sounds.experimentsClass.element_masking import mask_and_latent_BalancedNegatives
    mask_and_latent_BalancedNegatives(str(output_dir))
