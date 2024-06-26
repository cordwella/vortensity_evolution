# Gap formation driven by planets in globally isothermal invisicd discs

This repository implements the methods of Cimerman & Rafikov (2021),
Cimerman & Rafikov (2023) and Cordwell & Rafikov (2024) to generate
synthetic surface densities of gap formed by a planet in a 
protoplanetary discs. The 1D axisymmetric surface density of the 
protoplanetary disk can  be generated within seconds and provides
a good fit (especially in the outer disk) to the evolution of the disk.

Please see the associated notebooks for examples of function calls and 
output.

Running the python implementation requires matplotlib, 
numpy and scipy to be installed. 


## Example Output
![plot](example.png)

## Implementation Description
CR21 generated a semi-analytical formula for the size of the shock
generated by a sub thermal mass planet in a globally isothermal disk.
They then used this to describe the linear evolution in vortensity $\zeta$.
CR23 extended this to reconstruct the surface 
density of a disk from its vortensity.

CR24 derived a linear model of surface density evolution that takes in 
$f_{\text{dep}} = \Sigma^{-1} \partial F/\partial R$ as an input and 
estimates the future evolution. See the paper for details.

## Changes from the published papers
This repository fixes an error in the fitting parameters for $\Delta \chi$ 
in CR21. This was an error in the original paper due to a mistake in 
normalisiation, and is discussed further in CR24.

## Use in further work
These codes may be freely distributed and modified as long as 
credit to the author (Amelia J. Cordwell) remains. Any use 
in a scientific publication should cite this repository, 
Cimerman & Rafikov (2021) and Cordwell & Rafikov (2024).


## Referenced papers
Cimerman & Rafikov, 2021 - https://arxiv.org/abs/2108.01423
Cimerman & Rafikov, 2023 - https://arxiv.org/abs/2212.03062
Cordwell & Rafikov, 2024 - In prep.