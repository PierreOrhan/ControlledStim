ControlledStim is a python package to help generate auditory protocol that are then used as input in deep neural network.

ControlledStim is composed of the sounds package with subpackage:
- perExperiment:
  - protocols: this is where novel protocols are defined.
  - sequences: this is where sequence type shared across protocol are defined.
  - sound_elements: basic sound elements that compose sequence.
- experimentsClass:
    A set of tools useful to migrate generated dataset to a cluster. Also provide functionalities to add additional information to the generated dataset.

The goal of a ControlledStim protocol is to generate a set of sound file, indexed by a trials.csv document,
and accompanied by several supporting files.

Example of use:

        from sounds.perExperiment.protocols.AlRoumi2023 import RandRegRand_LOT
        from sounds.perExperiment.protocols.ProtocolGeneration import Protocol_independentTrial
        
        # Generate a default sequence from the LOT paradigm:
        r = RandRegRand_LOT() 
        # Create a protocol with indepent trial:
        p = Protocol_independentTrial(r) 
        # Generate the dataset from this protocol, here with one trial
        p.generate(n_trial=1,output_dir="your/output/dir")

Next we can add valuable information for the model that will be tested on these sequences.
For a self-supervised contrastive model, like Wav2vec2, one need to mask element of the sequences for which the loss will be computed.
We design ControlledStim to preserve as much information as possible on the sequence generated.
Downstream tool can therefore easily add information to the dataset by using the information already present there.
For example, to masks every element in a sequence successively (causal evaluation) one can use:

    
        from sounds.experimentsClass.element_masking import mask_and_latent_BalancedNegatives
        mask_and_latent_BalancedNegatives("your/output/dir")


We provide in sounds.tests several additional example.