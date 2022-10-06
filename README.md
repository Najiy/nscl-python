<<<<<<< HEAD
<<<<<<< HEAD
NSSN - Neuro-symbolic Spiking Network, NSSN

A pattern discovery algorithm with structurally dynamic networks.

    CLI mode:
        run main.py

    (see main.py for options)
    (save and load takes folder names - not filename)

Input to Spiketrain Encoding (Input datatype transformations):

    1. Booleans values to spiketrains 🗸
    2. Integers and Floating values to class-range spiketrains (k-ranges encoding) 🗸

Follow algorithm process flow:

    1. Structural - Generate inputs 🗸
    2. Feedinputs 🗸
    3. Propagate (update potentials t+=1) 🗸
    4. Structural - structural_plasticity (probationary) 🗸
    6. Functional - functional_plasticity 🗸
    7. TraceResults 🗸

SaveStates and ResumeStates 🗸

Hyperparameters:

    "BindingCount": 2,
    "PropagationLevels": 4,
    "PruneInterval": 30,
    "PruningThreshold": 0.2,
    "BindingThreshold": 0.5,
    "DownPotentialFactor": 0.2,
    
    "FiringThreshold": 0.25,
    "ZeroingThreshold": 0.05,
    "DecayFactor": 0.8,
    "InitialPotential": 0.2,
    "ReinforcementRate": 0.3,
    "RefractoryPeriod": 7,
    "PostSpikeFactor": 1.0,

    "DefaultEncoderCeiling": 1024,
    "DefaultEncoderFloor": 0,
    "DefaultEncoderResolution": 10,

    "TraceLength": 60


Forward and Backwards Prediction:

    1. static back-trace 🗸
    2. feed-forward unfolding prediction 🗸
