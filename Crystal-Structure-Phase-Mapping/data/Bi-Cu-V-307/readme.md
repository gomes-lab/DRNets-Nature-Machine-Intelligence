This directory contains input files for the Bi-Cu-V system, which are converted for use in DRNets, from the original instance file format in **../solutions/** . In the following, **P** is the number of pure phases provided in the ICDD library, **M** is the number of elements in the system, **N** is the number of data points, **Q** is the number of diffraction scattering vector magnitudes (angles) with measurements in each XRD pattern, **Q'** is the downsampled length, and **K** is the expected maximum number of phases present in the solution.

- **20k-bin-paths-len10.npy**, **200k-paths-len10-bin-paths-len10.npy**, **paths-len10**, **paths-len10-balanced** : a matrix of data point indices.  
	The pools of paths to choose from - each row is a random path through the composition graph. The different files have different numbers or distributions of paths.

- **bases_comp.npy** : a matrix of size (**P**, **M**), e.g., (100, 3) for Bi-Cu-V.  
    The elemental compositions of the possible pure phases in the ICDD library. 

- **bases_edge.npy** : a matrix of size (**P**, **P**), e.g., (100, 100) for Bi-Cu-V.  
    The similarity matrix of possible phases. If two phases are linked, then they are very similar to each other and can be considered interchangeable.

- **bases_name.npy** : a vector of size (**P**), e.g., (100) for Bi-Cu-V.  
    The names of the possible pure phases in the ICDD library.

- **bases_sol.npy** : an empty file (please ignore)

- **composition.npy** : a matrix of size (**N**, **M**), e.g., (307, 3) for Bi-Cu-V.  
    The elemental composition of the mixed material in each data point.

- **data.npy** : a matrix of size (**N**, **M**+**Q**), e.g., (307, 3 + 1197) for Bi-Cu-V.  
    For each row, the composition (length **M**) is concatenated with the XRD pattern (length **Q**), i.e. the XRD intensity at each diffraction angle.

- **degree_of_freedom.npy** : a vector of size (**N**).  
    The degrees of freedom of each data point, which is also the maximum possible number of co-existing pure phases at each data point.

- **edges.npy**, **edges_307.npy** : adjacency matrix of size (**N**, **N**) for the data points.  
    The edges are obtained by Delaunay triangulation in the composition space.

- **lib_comp.npy** : a matrix of size (**P**, **M**), e.g., (100, 3) for Bi-Cu-V.  
    The elemental compositions of the possible pure phases in the ICDD library. This is the same as bases_comp.npy.

- **lib_order.npy** : a vector of length (**P**).  
    Numerical indices for ICDD library entries (unused).

- **Q.npy** : a vector of length (**Q**).  
    The XRD scattering vector magnitudes (angles) for the XRD patterns.

- **Q_XXX.npy** : a vector of length (**Q'**).  
    The downsampled XRD scattering vector magnitudes (angles) for lower resolution versions of the XRD patterns.

- **Q_idx_300.npy** : a vector of length (**Q'**).  
    The indices of the downsampled XRD scattering vector magnitudes (angles) in the original list.

- **real_lib_comp.npy** : a matrix of size (**P**, **M**), e.g., (159, 3) for Bi-Cu-V.  
    The elemental compositions of the possible pure phases in the ICDD library. This is the same as bases_comp.npy.

- **sample_indicator.npy** : a matrix of size (**N**, **P**).  
	Each row is an indicator mask of possible phases at each data point, e.g., if the mixed material doesn't have element Bi, there can't be a phase containing Bi.

- **stick_bases.npy** : an object array of size (**K**).  
    The list of peaks (Q, intensity, width) for each ground truth phase. Only available for Bi-Cu-V (unused).

- **sticks_lib.npy** : an object array of size (**P**).  
    The stick pattern or list of peaks (Q, intensity) for each ICDD library phase.

- **XRD.npy** : a matrix of size (**N**, **Q**) 
	The unnormalized XRD patterns for each data point. 

- **weights_sol.npy** : a matrix of size (**N**, **M**) 
	The numpy format of the solution obtained from DRNets

