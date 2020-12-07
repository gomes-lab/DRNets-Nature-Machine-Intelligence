This directory contains input files for the Li-Sr-Al powder system, which are converted for use in DRNets, from the source data obtained from the author of the article "A deep-learning technique for phase identification in multiphase inorganic compounds using synthetic XRD powder patterns".
 In the following, **P** is the number of pure phases provided in the ICDD library, **M** is the number of elements in the system, **N** is the number of data points, **Q** is the number of diffraction scattering vector magnitudes (angles) with measurements in each XRD pattern, **Q'** is the downsampled length, and **K** is the expected maximum number of phases present in the solution.

- **20k-bin-paths-len5.npy** : a matrix of data point indices.  
	The pools of paths to choose from - each row is a random path through the composition graph. The different files have different numbers or distributions of paths.

- **bases_comp.npy** : a matrix of size (**P**, **M**), e.g., (37, 3) for Li-Sr-Al.  
    The elemental compositions of the possible pure phases in the ICDD library. 

- **bases_edge.npy** : a matrix of size (**P**, **P**), e.g., (37, 37) for Li-Sr-Al.  
    The similarity matrix of possible phases. If two phases are linked, then they are very similar to each other and can be considered interchangeable.

- **bases_name.npy** : a vector of size (**P**), e.g., (37) for Li-Sr-Al.  
    The names of the possible pure phases in the ICDD library.

- **composition.npy** : a matrix of size (**N**, **M**), e.g., (50, 3) for Li-Sr-Al.  
    The elemental composition of the mixed material in each data point.

- **data.npy** : a matrix of size (**N**, **M**+**Q**), e.g., (50, 3 + 4501) for Li-Sr-Al.  
    For each row, the composition (length **M**) is concatenated with the XRD pattern (length **Q**), i.e. the XRD intensity at each diffraction angle.

- **degree_of_freedom.npy** : a vector of size (**N**).  
    The degrees of freedom of each data point, which is also the maximum possible number of co-existing pure phases at each data point.

- **edges.npy**: adjacency matrix of size (**N**, **N**) for the data points.  
    The edges are obtained by Delaunay triangulation in the composition space.

- **gt_weights.npy** : a matrix of size (**N**, 3).  
    The ground-truth phase concentration of this Li-Sr-Al powder system, which is used to compute the prediction accuracy. Note that this system only contains 3 pure phases (#14, #36, #7).

- **lib_comp.npy** : a matrix of size (**P**, **M**), e.g., (37, 3) for Li-Sr-Al.  
    The elemental compositions of the possible pure phases in the ICDD library. This is the same as bases_comp.npy.

- **lib_order.npy** : a vector of length (**P**).  
    Numerical indices for ICDD library entries (unused).

- **Q.npy** : a vector of length (**Q**).  
    The XRD scattering vector magnitudes (angles) for the XRD patterns.

- **Q_idx.npy** : a vector of length (**Q**).  
    The indices of the XRD scattering vector magnitudes (angles) in the original list.

- **real_lib_comp.npy** : a matrix of size (**P**, **M**), e.g., (37, 3) for Li-Sr-Al.  
    The elemental compositions of the possible pure phases in the ICDD library. This is the same as bases_comp.npy.

- **sample_indicator.npy** : a matrix of size (**N**, **P**).  
	Each row is an indicator mask of possible phases at each data point, e.g., if the mixed material doesn't have element Li, there can't be a phase containing Li.

- **sticks_lib.npy** : an object array of size (**P**).  
    The stick pattern or list of peaks (Q, intensity) for each ICDD library phase.

