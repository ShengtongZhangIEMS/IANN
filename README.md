# IANN
This is a black-box function visualization package for Interpretable Architecture Neural Network (IANN)[1]

The basic idea of IANN is to approximate the black-box function $f$ by

```f(\mathbf{x}) = g(x_j, h(x_{\j}),```

for some input $x_j$, and two continuous functions $g$ and $h$. To visualize the effect of $x_j$, one can construct a 3D plot with $f$ vs. $x_j$ and $h$. Then, the IANN proceeds in a hierarchical way to further approximate $h$ with a similar structure.

It has two specific structure, each of which can be efficiently represented by IANN, OVH and DASH.   

**Original Variable Hierarchical (OVH):** visualize the effects of the original variables on the response $f$.   
![image](./paper_fig/OVH_eg.png)  


**Disjoint Active Subspace Hierarchical (DASH):** construct groups of disjoint linear combinations of the inputs and visualize the effects on $f$ with the groups as the inputs.  


[1] Zhang, Shengtong, and Daniel W. Apley. "Interpretable Architecture Neural Networks for Function Visualization." Journal of Computational and Graphical Statistics just-accepted (2023): 1-21.

# BIANN
This is a generalized version of IANN with a balanced split in the dichtomous tree structure.
