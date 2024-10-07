ControlledStim is a python package to help generate auditory protocol that are then used as input in deep neural network.

ControlledStim is composed of the sounds package with subpackage:
- perExperiment:
  - protocols: this is where novel protocol are defined
  - sequences: this is where sequence type shared across protocol are defined
  - sound_elements: basic sound elements that compose sequence.
- experimentsClass:
    A set of tools useful to migrate generated dataset to a cluster. Also provide functionalities to add additional information to the generated dataset.

The goal of a ControlledStim protocol is to generate a set of sound file, indexed by a trials.csv document,
and accompanied by several supporting files.

Example of use:
    'from sounds.perExperiment.protocols.AlRoumi2023 import RandRegRand_LOT 
    r = RandRegRand_LOT()'
    
    