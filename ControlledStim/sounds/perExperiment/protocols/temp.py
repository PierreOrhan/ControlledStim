
RFRAM_key = None
Bip_Rand_key = None
test_dir = str(pathlib.Path(__file__).parent.parent.absolute() / "tests")
data_dir = str(pathlib.Path(__file__).parent.parent.absolute() / "data")


samplerates_sound = {
    "gaussian_N": 16000,
    "gaussian_RN": 16000,
    "gaussian_RefRN": 16000,
    "bip_": 16000,
    "silence": 16000,
    "Bip_Rand": 16000,
    "Bip_Ref_": 16000
}

durations_sound = {
    "gaussian_N": 1000,
    "gaussian_RN": 1000,
    "gaussian_RefRN": 1000,
    "bip_": 1000,
    "silence": 1000,
    "Bip_Rand": 50,
    "Bip_Ref_": 50
}
consines_rmp_length = {
    "gaussian_N": 5,
    "gaussian_RN": 5,
    "gaussian_RefRN": 5,
    "bip_": 5,
    "silence": 5,
    "Bip_Rand": 5,
    "Bip_Ref_": 5
}

from datasets import load_dataset, Audio, Dataset


def rand(nb_elements: int) -> list[str]:
    sequence = nb_elements*["Bip_Rand"]
    return sequence

def reg(nb_cyc: int, Rcyc:int) -> list[str]:
    sequence = []
    for i in range(Rcyc):
        sequence.append("Bip_Ref_"+str(i+1))
    sequence *= nb_cyc
    return sequence

def rfram_sequence_exp1() -> list[str]:
    """Generates a correct RFRAM_1 sequence.

    :return: list[str]
    """

    nbN = 50
    nbRN = 100
    nbRefRN = 50
    N = "gaussian_N"
    RN = "gaussian_RN"
    refRN = "gaussian_RefRN"

    sequence = [N] * nbN + [RN] * nbRN + [refRN] * nbRefRN
    random.shuffle(sequence)

    for i in range(1, len(sequence)):
        if sequence[i] == refRN and sequence[i - 1] == refRN:
            while sequence[i] == refRN:
                j = random.randint(0, len(sequence) - 1)
                if 0 < j < len(sequence) - 1:
                    if sequence[j] != refRN and sequence[j + 1] != refRN and sequence[j - 1] != refRN:
                        sequence[i], sequence[j] = sequence[j], sequence[i]
                elif j == 0:
                    if sequence[j] != refRN and sequence[j + 1] != refRN:
                        sequence[i], sequence[j] = sequence[j], sequence[i]
                else:
                    if sequence[j] != refRN and sequence[j - 1] != refRN:
                        sequence[i], sequence[j] = sequence[j], sequence[i]

    return sequence


sequence_name_to_sounds = {
    "LOT_repeat": [],
    "LOT_alternate": [],
    "LOT_pairs": [],
    "LOT_quadruplets": [],
    "LOT_PairsAndAlt1": [],
    "LOT_Shrinking": [],
    "LOT_PairsAndAlt2": [],
    "LOT_threeTwo": [],
    "LOT_centermirror": [],
    "LOT_complex": [],
    # "tree1":tree1,
    # "tree2":tree2,

    "LocalGlobal_Standard": ["bip_1", "bip_1", "bip_1", "bip_1"],
    "LocalGlobal_Deviant_1": ["bip_1", "bip_1", "bip_1", "bip_2"],
    "LocalGlobal_Deviant_2": ["bip_1", "bip_1", "bip_1", "bip_3"],
    "LocalGlobal_Omission": ["bip_1", "bip_1", "bip_1", "silence"],

    "RandReg_5": [],
    "RandReg_8": [],
    "RandReg_10": [],
    "RandReg_20": []
}

protocol_name_to_sequences = {
    "LocalGlobal_ssss": ["LocalGlobal_Standard", "LocalGlobal_Standard", "LocalGlobal_Standard", "LocalGlobal_Deviant_1"],
    "LocalGlobal_sssd": ["LocalGlobal_Deviant_1", "LocalGlobal_Deviant_1", "LocalGlobal_Deviant_1", "LocalGlobal_Deviant_2"],
    "LocalGlobal_sss_": ["LocalGlobal_Deviant_1", "LocalGlobal_Deviant_1", "LocalGlobal_Deviant_1", "LocalGlobal_Omission"],
    "RandomRegular": [],
    "SyllableStream": [],
    "Habituation": [],
    "TestRandom": [],
    "TestDeviant": []
}

