import sounds.perExperiment.ProtocolGeneration as PG
import sounds.experimentsClass.element_masking as EM
import pathlib

data_dir = str(pathlib.Path(__file__).parent.parent.absolute() / "data")




soundpool = PG.Sound_pool("SP1")

seq1 = PG.Sequence("RFRAM_1", 0.5)
seq2 = PG.Sequence("LocalGlobal_Deviant_1",0.5)
seq3 = PG.Sequence("LocalGlobal_Deviant_2",0.5)
seq4 = PG.Sequence("LocalGlobal_Standard",0.5)
seq5 = PG.Sequence("LocalGlobal_Omission",0.5)

soundpool.addSequence(seq1)
soundpool.addSequence(seq2)
soundpool.addSequence(seq3)
soundpool.addSequence(seq4)
soundpool.addSequence(seq5)

comb = PG.Combinator("Dataset_1", 16000)
comb.combine([seq1,seq2,seq3,seq4,seq5], soundpool)

protocol_1 = PG.Protocol("LocalGlobal_sssd", 0.5)
ann_dataset_1 = protocol_1.create(str(pathlib.Path(data_dir) / comb.name / pathlib.Path("sequences.csv")))

masker = EM.ElementMasking("EM1")
dataset = masker._mask_and_latent(str(pathlib.Path(data_dir) / comb.name))
print(dataset)


