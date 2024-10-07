from pathlib import Path
from sounds.perExperiment.protocols.Bekinschtein2009 import RandRegDev_LocalGlobal
from sounds.perExperiment.protocols.AlRoumi2023 import RandRegRand_LOT_deviant
from sounds.perExperiment.protocols.Barascud2016 import RandRegRand
from sounds.perExperiment.protocols.ProtocolGeneration import ListProtocol_independentTrial
from sounds.perExperiment.sequences.lot_patterns import lot_patterns
from sounds.perExperiment.sound_elements import Bip
from sounds.perExperiment.sound_elements import Sound_pool
import numpy as np
from sounds.experimentsClass.element_masking import mask_and_latent_BalancedNegatives

# Example code to generate several protocols.

# We beging by generating a common sound pool to all protocols:
# To allow better comparison across paradigms, the random sequence is sampled but shared across protocols:
tones_fs=np.logspace(np.log(222),np.log(2000),20,base=np.exp(1))
sounds_rand = [Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f]) for idf, f in
          enumerate(np.array(tones_fs[:16])[[3, 11, 14,  2,  7,  9,  1, 13,  8,  5,  6,  0, 12,  4, 10, 15]])]
s_rand = Sound_pool.from_list(sounds_rand)

out_path = Path("your/output/dir")

# Generation of the local-global paradigm
output_dir = out_path / "localglobal"
motif_repeat = 3
low_freq = 222
high_freq = 2000
rs = []
# We explore several parametrization of the experiment: with increasing delay between tones and delay between sequence repetition:
for isi in np.arange(0,0.6,step=0.05):
    for seq_isi in [0.0,0.5,0.75]:
        # The twoo type of sequence:
        for k in ["localstandard","localdeviant"]:
            # generat one sequence:
            s_reg = Sound_pool.from_list([Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f])
                                          for idf, f in enumerate([low_freq, high_freq])])
            rs += [RandRegDev_LocalGlobal(name=k+"_globalDeviant"+"_highlow_isi-"+str(isi)+"_seqisi-"+str(seq_isi),
                                       global_standard=k,
                                       tones_fs=tones_fs,
                                       motif_repeat=motif_repeat,
                                       isi=isi,sequence_isi=seq_isi)]
            rs[-1].fixPoolSampled(s_rand, s_reg)

            # and the complementary sequence:
            s_reg = Sound_pool.from_list([Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=[f])
                                          for idf,f in  enumerate([high_freq,low_freq])])
            rs += [RandRegDev_LocalGlobal(name=k+"_globalDeviant"+"_lowhigh_isi-"+str(isi)+"_seqisi-"+str(seq_isi),
                                       global_standard=k,
                                       tones_fs=tones_fs,
                                       motif_repeat=motif_repeat,
                                       isi=isi,sequence_isi=seq_isi)]
            rs[-1].fixPoolSampled(s_rand, s_reg)
            # fixPoolSampled make sure that s_rand and s_reg are preserved and not resampled.
# Once all possible sequences qre collected, we group them with a ListProtocol:
lp = ListProtocol_independentTrial(rs)
# and generate them:
lp.generate(n_trial=1,output_dir=output_dir)
# additionally we save masks and negatives sampling for the study of Wav2vec2 contrastive loss:
mask_and_latent_BalancedNegatives(str(output_dir))


## Focus on the oddball paradigm:
output_dir = Path("/media/pierre/NeuroData2/datasets/lot_further/effectSilence") / "MMN"
rs = []
for isi in np.arange(0,0.6,step=0.05):
    for seq_isi in [0.0, 0.5, 0.75]:
        rs += [RandRegDev_LocalGlobal(name="MMN_isi-"+str(isi)+"_seqisi-"+str(seq_isi),
                                   global_standard="localstandard",
                                   tones_fs=tones_fs,
                                   motif_repeat=motif_repeat,
                                   isi=isi,sequence_isi=seq_isi)]
        rs[-1].s_rand = s_rand
lp = ListProtocol_independentTrial(rs)
lp.generate(n_trial=10,output_dir=output_dir)
mask_and_latent_BalancedNegatives(str(output_dir))


# Language of Thought paradigm (Al Roumi et al.):
low_freq = 222
high_freq = 2000
output_dir = out_path/ "LOT"
randSeqEnd = None
rs = []
for idtrial,(low_freq,high_freq) in enumerate([([350,700,1400],[500,1000,2000]),([800],[1600])]): #([222],[2000]),
    for k in lot_patterns.keys():
        for isi in np.arange(0, 0.6, step=0.05):
            for seq_isi in [0.0, 0.5, 0.75]:
                s_reg = Sound_pool.from_list([Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=f)
                                              for idf,f in  enumerate([low_freq,high_freq])])
                rs += [RandRegRand_LOT_deviant(name=k+"_isi-"+str(isi)+"_seqisi-"+str(seq_isi)+"_lowhigh_trial"+str(idtrial),
                                           lot_seq=k,
                                           tones_fs=tones_fs,
                                           motif_repeat=motif_repeat,
                                           isi=isi,sequence_isi=seq_isi,
                                           deviant=0)]
                rs[-1].fixPoolSampled(s_rand, s_reg)

                s_reg = Sound_pool.from_list([Bip(name="bip-" + str(idf), samplerate=16000, duration=0.05, fs=f)
                                              for idf,f in  enumerate([high_freq,low_freq])])
                rs += [RandRegRand_LOT_deviant(name=k+"_isi-"+str(isi)+"_seqisi-"+str(seq_isi)+"_highlow_trial"+str(idtrial),
                                           lot_seq=k,
                                           tones_fs=tones_fs,
                                           motif_repeat=motif_repeat,
                                           isi=isi,sequence_isi=seq_isi,
                                           deviant=0)]
                rs[-1].fixPoolSampled(s_rand, s_reg)

lp = ListProtocol_independentTrial(rs)
lp.generate(n_trial=1,output_dir=output_dir)
mask_and_latent_BalancedNegatives(str(output_dir))


# RandRegRand as in Barascud et al:
output_dir = out_path  / "Barascud"
tones_fs=np.logspace(np.log(222),np.log(2000),20,base=np.exp(1))
rs = []
# for seq_isi in [0.5]: #np.arange(0,0.5,step=0.1):
for isi in np.arange(0, 0.6, step=0.05):
    for nbmotif,cycle in zip([7,3],[5,10]):
        rs += [RandRegRand(name="isi-"+str(isi)+"_cycle"+str(cycle),
                           tones_fs=tones_fs,
                           motif_repeat=nbmotif,
                           cycle = cycle,
                           isi=isi,sequence_isi=0)]
lp = ListProtocol_independentTrial(rs)
lp.generate(n_trial=15,output_dir=output_dir) #15
from sounds.experimentsClass.element_masking import mask_and_latent_BalancedNegatives
mask_and_latent_BalancedNegatives(str(output_dir))