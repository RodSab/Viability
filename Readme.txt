File Viab_generic_example_Via_BEE_lity_stoch0.13.py is a code written in Python that may be used to solve some basic viability problems. It is developped for discreate time problems. It can be found here:

https://github.com/RodSab/Viability/blob/main/Viab_generic_example_Via_BEE_lity_stoch_0.13.py

The phylosophy of this code is to be simple, easy to reuse and easy to adapt. It is not optimized and is rather suitable if you want to have a first overview of a simple viability related algorithm and if you have a low dimension viability problem.

It computes :
    - The viability kernel
    - Resilience bassins
    - Maps of adaptability and robustness

It currently works for up to 3 states and 5 controls.

The model is illustrated here with a case study related to the sustainability of a bee farming system 
see Kouchner et al. (2019) for details on the case study:
 - Kouchner C. et al. (2019) Intégrer l’adaptabilité dans l’analyse de la durabilité des exploitations apicoles, Innovations Agronomiques (77)31-43
 https://hal.inrae.fr/hal-02900352/document


This code is ment to be quite generic and is illustrated here with an example that should be replaced by the user's case study.

User should adjust the following sections to its own case study:
    - General parameters
    - Case study related parameters
    - Case study related functions
    - Constraints
    - Dynamics of the system    

Plotting functions are rather specific to the acse study although they could eaisily be adapted to a problem of similar size. 
    