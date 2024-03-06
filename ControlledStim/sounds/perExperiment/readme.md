Organisation of the code for the project.

We want to generate huggingface dataset containing a set of sounds.
The dataset will also contain additional information about the sounds 
for the different experimental protocol and for the different neural networks.
For example: masks over latent tokens (downsampled) for Wav2vec2

Sounds can take the following formats:
- Discrete stream:
    Each sounds is composed of independently synthesized sound element.
- Continuous stream:
    Each sound is composed of jointly synthesized sound element.

Pipelines:
   1) Generation of the set of sequences.
      - Goal 1: different generator for different types of sounds.
      - Goal 2: different sequences.
      - Goal 3: combinators of sequences and sounds.
           The combinator should be parametrizable by a set of parameters.
      - Results:
           - Generation of a set of Sequence
           - Generation of a set of Sound
           - Generation of a SoundSequence_Dataset which contain all sequences of sounds.
   2) Embedding in protocol.
      - Goal 1: define the different set of protocol 
        (RandReg, Syllable Stream, Habituation, Test-Random, Test-Deviant, ...)
           The protocol should be parametrizable by a set of parameters.
           The protocol should combine sequences together.
      
      - Results: Generation of a Protocol_Dataset
   3) Adding of the specific elements for each neural network.
      - Goal 1: define sets of specific embedding for each neural network
      - Goal 2: identify classes of networks that verify the same embeddings.
      - Results: Generation of an ANN_Dataset 
   
Sequence_Dataset:
    The sequence dataset should contain all annotations necessary for the ANN_dataset to perform their analyses.

Point of attentions:
    a) Sounds will be grouped to a pool of sounds, with certain criteria to be respected when combined as sequences.
    b) For RandReg: progressive diminution of the pool.
    c) ISI: time during two sound element in a sequence
    d) Sequence_ISI: time that separate two sequences in the protocol.
all possible sequences could be stored in .txt files

Difficulty: 
    we need to leave the information of what is where for the protocol. 

    For example:
        1) In the local global protocol, the protocol can be:
            regular regular .... regular ... deviant 
            Where regular will be either ssss or sssd
        2) In the SyllableChunk protocol, the protocol can be:
            word1 word3 word2 word1 word1 .... 
            we alternate randomly between the words. 
    
    We can't just keep the sounds array, as for the masking, we will need to mask 
    individual sound elements independently.
    We therefore have to preserve where everything is in the sound.

Classes:

    Sequence:
        name: str
        algebra: List[int]
        isi: float (in seconds)

    Sound:
        name: str
        sound: np.ndarray
        sample_rate: int

    Sound_pool:
        name: str
        sounds: dict[Sound]

    Combinator:
        name: str
        def combine(seq_list: List[Sequence],sound_pool: Sound_pool) -> SoundSequence_Dataset:

    Protocol:
        name: str
        sequence_isi: float (in seconds)
        def create(soundseq_dataset : SoundSequence_Dataset) -> ANN_Dataset

            

naming convention:
    Protocol_Combinator_Sound_