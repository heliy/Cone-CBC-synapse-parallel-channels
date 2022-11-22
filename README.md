# Linear-Nonlinear-Synapse model for synaptic depressions and temporal frequency tuning

This is repository accompanies the paper: 

Liuyuan He, Yutao He, Lei Ma, and Tiejun Huang (2022)

[A theoretical model reveals specialized synaptic depressions and temporal frequency tuning in retinal parallel channels](https://www.frontiersin.org/articles/10.3389/fncom.2022.1034446/full) 

# codes

- `lib`: library for RAB module, LN model, and LNS models.
- `Grabner.py`: Experiments on (Grabner et al., 2016)
- `Schroeder.py`: Experiments on (Schroeder et al., 2021)

# requirements

- Python3
- numpy
- matplotlib
- scipy

# experiments

## for (Grabner et al., 2016)

```
python3 Grabner.py
```

> Experiments on (Grabner et al., 2016)
> 
> Load Optimized Models
> 
> ##################################################
> ### The performance of LN models # Figure 1
> ##################################################
> 1. Response Traces # Figure 1A
> 2. Peak Responses # Figure 1B
> 3. Tuning ranges # Figure 1D
> 4. Parameters in LN models # Figure 1C
> 
> ##################################################
> ### The performance of LNS models # Figure 2
> ##################################################
> 1. Response Traces # Figure 2B
> 2. Peak Responses # Figure 2C
> 3. Tuning ranges # Figure 2D
> 
> ##################################################
> ### The synaptic depression experiment on LNS models # Figure 3
> ##################################################
> 1. Parameters of LNS models # Figure 3B
> 2. The PPD experment on two LNS models # Figure 3C&D
> 
> ##################################################
> ### Kinetics inside LNS models # Figure 4
> ##################################################
> 1. The inner variables of the cb2 LNS model # Figure 4A
> 2. The depression of inner variables of the cb2 LNS model # Figure 4B
> 3. The inner variables of the cb2 LNS model with time constants # Figure 4C
> 4. The inner variables of the cb3a LNS model with time constants # Figure 4D




## for (Schroeder et al., 2021)

```
python3 Schroeder.py
```

> Experiments on (Schroeder et al., 2016)
> 
> ##################################################      
> ### Data fitting and parameters of LNS models # Figure 5
> ##################################################      
> 1. Response Traces # Figure 5D
> 2. Parameters # Figure 5E
> 
> ##################################################
> ### Prediction of temporal tuning # Figure 6      
> ##################################################
> 1. Filtering responses # Figure 6A
> 2. Peak responses # Figure 6B