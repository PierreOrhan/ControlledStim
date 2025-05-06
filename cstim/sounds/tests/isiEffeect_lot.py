from pathlib import Path

from sounds.perExperiment.protocols.AlRoumi2023 import RandRegRand_LOT,RandRegRand_LOT_deviant,RandRegRand_LOT_orig
from sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial
from sounds.perExperiment.sequences.lot_patterns import lot_patterns
### Debugging: We fix the pool and repeat the probing over and over
import pandas as pd
from sounds.perExperiment.sound_elements import Bip,Silence
from sounds.perExperiment.sound_elements import Sound_pool,Sound
import numpy as np
import copy
#

output_dir = Path("/media/pierre/NeuroData2/datasets/lot_further/lot_ISIrepro") / "randregrandLONG"

tones_fs=np.logspace(np.log(222),np.log(2000),20,base=np.exp(1))
sounds_rand = [Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f]) for idf, f in
          enumerate(np.array(tones_fs[:16])[ [3, 11, 14,  2,  7,  9,  1, 13,  8,  5,  6,  0, 12,  4, 10, 15]])]
s_rand = Sound_pool.from_list(sounds_rand)

randSeqEnd = None
rs = []
for k in lot_patterns.keys():
    # for seq_isi in [0.5]: #np.arange(0,0.5,step=0.1):
    for isi in np.arange(0,1,step=0.1):#np.arange(0,0.3,step=0.05):
        # for deviant in [1]:
        s_reg = Sound_pool.from_list([Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f])
                                      for idf,f in  enumerate([222,2000])]) #889.8296701050009, 1413.4860237345383
        rs += [RandRegRand_LOT(name=k+"_lot_isi-"+str(isi)+"_lowhigh",
                                   lot_seq=k,
                                   tones_fs=tones_fs,
                                   motif_repeat=3,
                                   isi=isi,sequence_isi=0)]
        rs[-1].fixPoolSampled(s_rand, s_reg)
        if randSeqEnd == None:
            randSeqEnd = rs[-1].randSeqEnd
        else:
            rs[-1].randSeqEnd = copy.deepcopy(randSeqEnd)
            rs[-1].randSeqEnd.isi = isi

        s_reg = Sound_pool.from_list([Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f])
                                      for idf,f in  enumerate([2000,222])]) #889.8296701050009, 1413.4860237345383
        rs += [RandRegRand_LOT(name=k+"_lot_isi-"+str(isi)+"_highlow",
                                   lot_seq=k,
                                   tones_fs=tones_fs,
                                   motif_repeat=3,
                                   isi=isi,sequence_isi=0)]
        rs[-1].fixPoolSampled(s_rand, s_reg)
        rs[-1].randSeqEnd = copy.deepcopy(randSeqEnd)
        rs[-1].randSeqEnd.isi = isi

        # rs += [RandRegRand_LOT_deviant(name=k+"_lot_Deviant_isi-"+str(isi)+"_seqisi-"+str(seq_isi)+"_deviant-"+str(deviant),
        #                            lot_seq=k,
        #                            tones_fs=tones_fs,
        #                            deviant=deviant,
        #                            motif_repeat=10,
        #                            isi=isi,sequence_isi=seq_isi)]
        #

lp = ListProtocol_independentTrial(rs)
lp.generate(n_trial=1,output_dir=output_dir)
from sounds.experimentsClass.element_masking import mask_and_latent_BalancedNegatives
mask_and_latent_BalancedNegatives(str(output_dir))


#
# from sounds.perExperiment.protocols.Barascud2016 import RandRegRand
#
# output_dir = Path("/media/pierre/NeuroData2/datasets/lot_further/lot_ISIrepro") / "randregrandBarascudShort"
#
# tones_fs=np.logspace(np.log(222),np.log(2000),20,base=np.exp(1))
# rs = []
# # for seq_isi in [0.5]: #np.arange(0,0.5,step=0.1):
# for isi in np.arange(0,0.3,step=0.025):#np.arange(0,0.3,step=0.05): #0,0.3,step=0.025
#     for nbmotif,cycle in zip([7,3],[5,10]):
#         rs += [RandRegRand(name="isi-"+str(isi)+"_cycle"+str(cycle),
#                            tones_fs=tones_fs,
#                            motif_repeat=nbmotif,
#                            cycle = cycle,
#                            isi=isi,sequence_isi=0)]
#
# lp = ListProtocol_independentTrial(rs)
# lp.generate(n_trial=1,output_dir=output_dir) #15
# from sounds.experimentsClass.element_masking import mask_and_latent_BalancedNegatives
# mask_and_latent_BalancedNegatives(str(output_dir))
