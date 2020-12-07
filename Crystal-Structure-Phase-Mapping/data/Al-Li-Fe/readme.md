This directory contains input files for the Al-Li-Fe system, which are converted for use in DRNets, from the original instance file format in **../solutions/** . In the following, **P** is the number of pure phases provided in the ICDD library, **M** is the number of elements in the system, **N** is the number of data points, **Q** is the number of diffraction scattering vector magnitudes (angles) with measurements in each XRD pattern, **Q'** is the downsampled length, and **K** is the expected maximum number of phases present in the solution.

- **bases_comp.npy** : a matrix of size (**P**, **M**), e.g., (159, 3) for Al-Li-Fe.  
    The elemental compositions of the possible pure phases in the ICDD library. 

- **bases_name.npy** : a vector of size (**P**), e.g., (159) for Al-Li-Fe.  
    The names of the possible pure phases in the ICDD library.

- **bases_sol.npy** : an empty file (please ignore)

- **composition.npy** : a matrix of size (**N**, **M**), e.g., (231, 3) for Al-Li-Fe.  
    The elemental composition of the mixed material in each data point.

- **data.npy** : a matrix of size (**N**, **M**+**Q**), e.g., (231, 3 + 650) for Al-Li-Fe.  
    For each row, the composition (length **M**) is concatenated with the XRD pattern (length **Q**), i.e. the XRD intensity at each diffraction angle.

- **decomposition_sol.npy** : a tensor of size (**N**, **P**, **Q**), e.g., (231, 6, 650) for Al-Li-Fe.  
	The unnormalized ground truth solution for the decomposed XRD signals at each data point (only exists for Al-Li-Fe). This is unused - see the ground truth solution file in **../../solutions/** instead.

- **degree_of_freedom.npy** : a vector of size (**N**).  
    The degrees of freedom of each data point, which is also the maximum possible number of co-existing pure phases at each data point.

- **edges.npy** : adjacency matrix of size (**N**, **N**) for the data points.  
    The edges are obtained by Delaunay triangulation in the composition space.

- **lib_comp.npy** : a matrix of size (**P**, **M**), e.g., (159, 3) for Al-Li-Fe.  
    The elemental compositions of the possible pure phases in the ICDD library. This is the same as bases_comp.npy.

- **lib_order.npy** : a vector of length (**P**).  
    Numerical indices for ICDD library entries (unused).

- **normalized_weights_sol.npy** : a matrix of size (**N**, **K**), e.g., (231, 6) for Al-Li-Fe.  
	The ground truth solution of the phase concentration at each data point (only exists for Al-Li-Fe).

- **paths-len10.npy** : a matrix of data point indices.  
    The pool of paths to choose from - each row is a random path through the composition graph.

- **Q.npy** : a vector of length (**Q**).  
    The XRD scattering vector magnitudes (angles) for the XRD patterns.

- **Q_XXX.npy** : a vector of length (**Q'**).  
    The downsampled XRD scattering vector magnitudes (angles) for lower resolution versions of the XRD patterns.

- **Q_idx_650.npy** : a vector of length (**Q'**).  
    The indices of the downsampled XRD scattering vector magnitudes (angles) in the original list.

- **real_lib_comp.npy** : a matrix of size (**P**, **M**), e.g., (159, 3) for Al-Li-Fe.  
    The elemental compositions of the possible pure phases in the ICDD library. This is the same as bases_comp.npy.

- **sample_indicator.npy** : a matrix of size (**N**, **P**).  
	Each row is an indicator mask of possible phases at each data point, e.g., if the mixed material doesn't have element Al, there can't be a phase containing Al.

- **stick_bases.npy** : an object array of size (**K**).  
    The list of peaks (Q, intensity, width) for each ground truth phase. Only available for Al-Li-Fe (unused).

- **sticks_lib.npy** : an object array of size (**P**).  
    The stick pattern or list of peaks (Q, intensity) for each ICDD library phase.

- **unnormalized_weights_sol.npy** : a matrix of size (**N**, **K**), e.g., (231, 6) for Al-Li-Fe.  
	The unnormalized phase concentration at each data point in the ground truth solution. Only provided for Al-Li-Fe (unused).

- **weights_sol.npy** : a matrix of size (**N**, **K**), e.g., (231, 6) for Al-Li-Fe.  
	The unnormalized phase concentration at each data point in the ground truth solution. Only provided for Al-Li-Fe (unused). This is the same as unnormalized_weights_sol.npy.

- **XRD.npy** : a matrix of size (**N**, **Q**)  
	The unnormalized XRD patterns for each data point. 


