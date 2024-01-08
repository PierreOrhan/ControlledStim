import os

import numpy as np
from pathlib import Path

from sounds.perExperiment.protocols.AlRoumi2023 import RandRegRand_LOT,RandRegRand_LOT_Generalize,RandRegRand_LOT_deviant,\
    RandRegRand_LOT_deviant_BoundFixedPool,RandRegRand_LOT_deviant_fixedPool,RandRegRand_LOT_fixedPool
from sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial

output_dir = Path("/media/pierre/NeuroData2/datasets/lot_testGeneration") / "test_randregrand"

# r1= RandRegRand_LOT(lot_seq="pairs",tones_fs=np.logspace(np.log(50),np.log(400),num=50),
#                    motif_repeat=10,isi=0,sequence_isi=0.2)
# for seq_isi in np.arange(0,0.2,step=0.05):
#     for isi in np.arange(0,0.3,step=0.05):
# seq_isi = 0.5
# isi = 0.2

### Debugging: We fix the pool and repeat the probing over and over
import pandas as pd
from sounds.perExperiment.sequences import lot_patterns,ToneList,Sequence,RandomPattern
from sounds.perExperiment.sound_elements import Bip,Silence
from sounds.perExperiment.sound_elements import Sound_pool,Sound
from sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
from sounds.perExperiment.sound_elements import ramp_sound,normalize_sound
from dataclasses import dataclass,field
import numpy as np
from typing import Union,Tuple

tones_fs=np.logspace(np.log(222),np.log(2000),20,base=np.exp(1))
sounds_rand = [Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f]) for idf, f in
          enumerate(np.array(tones_fs[:16])[ [3, 11, 14,  2,  7,  9,  1, 13,  8,  5,  6,  0, 12,  4, 10, 15]])]
s_rand = Sound_pool.from_list(sounds_rand)

rs = []
for idfreq_high,freq_high in enumerate(tones_fs[::2]):
    for idfreq_low, freq_low in enumerate(tones_fs[::2]):
        s_reg = Sound_pool.from_list([Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f])
                                      for idf,f in  enumerate([freq_low,freq_high])]) #889.8296701050009, 1413.4860237345383
        seq_isi = 0
        isi = 0
        rs += [RandRegRand_LOT_deviant_BoundFixedPool(name="RandReg_lot_Deviant_freqHigh"+str(idfreq_high)+"_freqLow"+str(idfreq_low), #isi-"+str(isi)+"_seqisi-"+str(seq_isi)+"_deviant-"+str(0)
                                   lot_seq="pairs",
                                   tones_fs=tones_fs,
                                   deviant=0,
                                   motif_repeat=3,
                                   isi=isi,sequence_isi=seq_isi)]
        rs[-1].fixPoolSampled(s_rand, s_reg)

        rs += [RandRegRand_LOT_fixedPool(name="RandReg_lot_Orig_freqHigh"+str(idfreq_high)+"_freqLow"+str(idfreq_low), #isi-"+str(isi)+"_seqisi-"+str(seq_isi)+"_deviant-"+str(0)
                                   lot_seq="pairs",
                                   tones_fs=tones_fs,
                                   motif_repeat=3,
                                   isi=isi,sequence_isi=seq_isi)]
        rs[-1].fixPoolSampled(s_rand, s_reg)

    # rs[0].samplePool(0,np.inf)
    #
    # ### Make a plot as a function of the distance between the two tones:
    # for deviant in np.arange(1,4):
    #     r = RandRegRand_LOT_deviant_BoundFixedPool(name="RandReg_lot_isi-"+str(isi)+"_seqisi-"+str(seq_isi)+"_deviant-"+str(deviant),
    #                            lot_seq="pairs",
    #                            tones_fs=np.logspace(np.log(222),np.log(2000),20,base=np.exp(1)),
    #                            deviant=deviant,
    #                            motif_repeat=6,
    #                            isi=isi,sequence_isi=seq_isi)
    #     r.fixPoolSampled(rs[0].s_rand,rs[0].s_reg)
    #     rs += [r]

lp = ListProtocol_independentTrial(rs)
lp.generate(n_trial=1,output_dir=output_dir)
from sounds.experimentsClass.element_masking import mask_and_latent_BalancedNegatives
mask_and_latent_BalancedNegatives(str(output_dir))
