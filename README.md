## Strategy from Structure: Individual Preferences in Decision Making Strategies Adaptively Reflect Differences in Brain Network Connectivity


### Decoding Decision-Making Strategies Through Resting-State fMRI Data   

Two dominant framworks exists to explain how people make decisions from experience. In the _Reinforcement Learning_ (RL) frameowork, decisions are progressively calibrated by adjusting the expected value of an option throuh differences in reward predictions. In the _Instance Based Learning_ (IBL) and the _Decision by Sampling_ frameworks, on the other hand, decisions are supposedly made by sampling from memory the outcomes of previous choices.

The two approaches are very similar and make almost identical predictions, so much so that scientists often refer to them interchangeably. These two frameoworks, however entail different interpretations in the term of the neural circuitry driving the decision process.

In RL, the expected utilities of previous choices are stored are cached values associated with each option. These values do not decay over time, and the association between option and value is supposed to be learned through dopaminergic reinforcement of the basal ganglia circuitry. This suggests a procedural learning system.

In IBL, the expected utilities are stored as episodic memories, and decisions are made by either averaging over their values of estimating a mean by repeatedly sampling from memory. In this case, values are associated with each episodic trace, and the global value of an option is shaped by the previous history as well as the rules that shape memory forgetting. This framewro implies that decision-making is carried out by the hippocampal-prefrontal neural circuitry that underpins declarative memory encoding and retrieval.

Furthermore, it is possible that different individuals behave differently, and might rely on one circuit or another. If so, it would make sense that different idividuals play to their own specific strengths, and use the system that gives the best results, given their neural makeup.

### Distinguishing IBL from RL

Because IBL-based decisions relies on decaying traces of previous choices, it makes subtly different predictions than RL. To distinguish the two, we will model decision-making using an RL or IBL framework in ACT-R (an integrated cognitive architecture that allows to model both). Each model is fit to each individual, and the model whose parameters yield the best match (using Log Likelihood) will be taken as evidence of the decision-making strategy used.

### The Task

As a task, we are using the "Incentive Processing" task of the Human Connectome Project. Data from N=199 participants, both the behavioral data and resting-state fMRI is available. 

<img src="https://docs.google.com/drawings/d/e/2PACX-1vS373PoD0nMc6mTr8kmLat6fbGaMgn2-ieAMiHCi11yYMraAO7BfZGrMErdditB9YP-zlC6DBsqs6fJ/pub?w=350&amp;h=187">

### The Model

Two distinct models: **Declarative Model** and **Procedural Model** are built in ACT-R to represent two different strategy selection preferences. See paper about the details of model 

**Declarative Model**

<img src="https://docs.google.com/drawings/d/e/2PACX-1vRBejRO2gF2IfEnHJeLZJ18ziZwVaLjcwTzvfoNGvQgAnYsPkvTRys9qXzzsmgCR66V4ajeUUuyWNgj/pub?w=500">

**Procedural Model**

<img src="https://docs.google.com/drawings/d/e/2PACX-1vRe2y7P9O1r3viqCgBXPwpS_zct9VYva5SWg0VaqrIRfReu0goFNqNfsmnym6eBekoxo-TfnkLlEYQi/pub?w=500">


### Model simulation & analysis

- Behavioral data are in `behavior`

- Model simulation data are in `actr-model`

- Resting-state fMRI data analysis is in  `rfmri`

- Task-based fMRI data analysis is in `tfmri`










