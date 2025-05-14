### Table of contents
- [Report](#report)
- [Selection of the prior distributions](#selection-of-the-prior-distributions)
- [Introduction to the CCT model](#introduction-to-the-cct-model)
- [Results](#results)

# Report
### Selection of the prior distributions
I chose the Beta distribution (scaled to [0.5, 1]) to represent the prior knowledge of the informants. This better aligns with the CCT assumptions that cultural knowledge is not distributed evenly across the population but instead that most people have moderate competence levels. This offers us more flexibility to encode varying degrees of knowledge. 
<br>
The Bernoulli(0.5) distribution represents to the prior probability of each item's consensus answer. This represents the uncertainty of the answer before data is observed. This allows the model to learn new data.

### Introduction to the CCT model
The Cultural Consensus Theory (CCT) is a model represents how knowledge is distributed across a population. We are not given the correct answer in advance and do not know who the experts are. Rather, it's based on the agreed answer that most people give. 

$$
p_{ij} = Z_j \cdot D_i + (1 - Z_j) \cdot (1 - D_i)
$$


- Each person is given a knowledge score before 50% (guessing) to 100% (expert). 
- Each question is formatted as a binary answer (Yes/No) with a true answer (i.e. the actual answer) that most knowledgeable person would give
- The likelihood of a correct answer is a function of the person's knowledge score
- The answer with the largest majority is assumed to be correct (consensus answer)