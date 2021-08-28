# Decoding Decision-Making Strategies Through Resting-State fMRI Data   

Two dominant framworks exists to explain how people make decisions from experience. In the _Reinforcement Learning_ (RL) frameowork, decisions are progressively calibrated by adjusting the expected value of an option throuh differences in reward predictions. In the _Instance Based Learning_ (IBL) and the _Decision by Sampling_ frameworks, on the other hand, decisions are supposedly made by sampling from memory the outcomes of previous choices.

The two approaches are very similar and make almost identical predictions, so much so that scientists often refer to them interchangeably. These two frameoworks, however entail different interpretations in the term of the neural circuitry driving the decision process.

In RL, the expected utilities of previous choices are stored are cached values associated with each option. These values do not decay over time, and the association between option and value is supposed to be learned through dopaminergic reinforcement of the basal ganglia circuitry. This suggests a procedural learning system.

In IBL, the expected utilities are stored as episodic memories, and decisions are made by either averaging over their values of estimating a mean by repeatedly sampling from memory. In this case, values are associated with each episodic trace, and the global value of an option is shaped by the previous history as well as the rules that shape memory forgetting. This framewro implies that decision-making is carried out by the hippocampal-prefrontal neural circuitry that underpins declarative memory encoding and retrieval.

Furthermore, it is possible that different individuals behave differently, and might rely on one circuit or another. If so, it would make sense that different idividuals play to their own specific strengths, and use the system that gives the best results, given their neural makeup.

## Distinguishing IBL from RL

Because IBL-based decisions relies on decaying traces of previous choices, it makes subtly different predictions than RL. To distinguish the two, we will model decision-making using an RL or IBL framework in ACT-R (an integrated cognitive architecture that allows to model both). Each model is fit to each individual, and the model whose parameters yield the best match (using Log Likelihood) will be taken as evidence of the decision-making strategy used.

## The Task

As a task, we are using the "Incentive Processing" task of the Human Connectome Project. Data is collected from N=176 participants, for whom both the behavioral data and resting-state fMRI is available. 






