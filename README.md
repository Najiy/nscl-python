Motivational scenario

Contribution

Input to Spiketrain Encoding (Input datatype transformations):

    1. Booleans values to spiketrains
    2. Integers and Floating values to class-range spiketrains (k-ranges encoding)

Follow algorithm process flow:

    1. Structural - Generate inputs 🗸
    2. Feedinputs 🗸
    3. Propagate (update potentials t+=1)
    4. Structural - GenerateNeurones (probationary) 🗸
    4.1 Limit to two preneurone connectivity
    5. Structural - GenerateSynapses (probationary) 🗸
    6. Functional - ReinforceSynapses
    7. TraceResults 🗸

SaveStates and ResumeStates 🗸

Evluation:

    1. Running times
    2. O-Complexity

Hyperparameters:

    1. neurone potential decay coefficient (G1)
        0.8 if len(n.fsynapses) == 0 else 0.5 + 1 / len(n.fsynapses) * 0.4
    2. neurone activation threshold (G2)
        if neurones[n].potential > 0.5
    3. on-new-neurone potential reduction coefficient for pre-neurones (G3)
        neurones[a].potential *= 0.5
    4. generate neurones with full potential (G4)
    5. reinforce formula (G5)
        wgt += (1 - wgt) * 0.5
    6. max propagation heirarchy (G6)
        for l in range(6):

Parallel Compute:

    1. Parallel potentials decay (taus)
    2. Parallel decay
    3. Compare heatmaps (results determines the approach to manifolds)

Forward and Backwards Prediction:

    1. static prediction (SP) (input neurones at t=0 only) -> sequences of inputs
    2. temporal prediction (TP) (input neurones allowing t=@ different times) -> sequences of inputs
    3. conditional pathway limiter for SP and TP
    4. descriptive (explainable-ai) sequences into words



BFS CRF BNet

Chapters:
    
    1. Joint Probability Distribution
    2. Temporally-induced Context


    Content: bayesian networks, conditional random fields, joint-probability distribution, temporally-induced context, dependency-pathway tracing, spiking neural networks, etc.