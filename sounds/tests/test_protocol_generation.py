import sounds.perExperiment.ProtocolGeneration as PG

def test_protocol_generation():

    soundpool = PG.Sound_pool("SP1")

    seq1 = PG.Sequence("RFRAM_1", 0.5)
    seq2 = PG.Sequence("LocalGlobal_Deviant_1",0.5)
    seq3 = PG.Sequence("LocalGlobal_Deviant_2",0.5)
    seq4 = PG.Sequence("LocalGlobal_Standard",0.5)
    seq5 = PG.Sequence("LocalGlobal_Omission",0.5)

    comb = PG.Combinator("Dataset_1", 16000)

    comb.combine([seq1,seq2,seq3,seq4,seq5], soundpool)
    return True

