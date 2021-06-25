# Deep Reasoning Networks (DRNets)

DRNets provide a general framework for integrating pattern recognition capabilities with reasoning about prior knowledge. The result is an end-to-end framework that can solve a broad range of mixed reasoning and learning problems. At a high level, DRNets combine deep learning with constraint reasoning by formulating a task as a data-driven constrained optimization problem, which is then reduced through a sequence of transformations into a data-driven unconstrained optimization problem amenable to end-to-end optimization via state-of-the-art deep learning technology.

## Main contributors

This code and data were primarily prepared by:

- Di Chen (Cornell University, Dept. of Computer Science), dc874@cornell.edu
- Yiwei Bai (Cornell University, Dept. of Computer Science), yb263@cornell.edu
- Sebastian Ament (Cornell University, Dept. of Computer Science), sea79@cornell.edu
- Wenting Zhao (Cornell University, Dept. of Computer Science), wz346@cornell.edu 
- Dan Guevarra (California Institute of Technology), guevarra@caltech.edu
- Lan Zhou (California Institute of Technology), lzhou@caltech.edu
- Bart Selman (Cornell University, Dept. of Computer Science), selman@cs.cornell.edu
- R. Bruce van Dover (Cornell University, Dept. of Materials Science and Engineering), vandover@cornell.edu
- John M. Gregoire (California Institute of Technology), gregoire@caltech.edu
- Carla P. Gomes (Cornell University, Dept. of Computer Science), gomes@cs.cornell.edu

## Citation

This work accompanies the following publication:

- Di Chen, Yiwei Bai, Sebastian Ament, Wenting Zhao, Dan Guevarra, Lan Zhou, Bart Selman, R. Bruce van Dover, John M. Gregoire, Carla P. Gomes. Automating Crystal-Structure Phase Mapping by Combining Deep Learning with Constraint Reasoning *Nature Machine Intelligence*, under review (2021).

## Directory structure

There is a top-level directory for each of the three example applications:

- **Crystal-Structure-Phase-Mapping** : Crystal structure phase mapping for two chemical systems: (1) a ternary Al-Li-Fe oxide system, which is theoretically-based, synthetically generated, with ground-truth solutions, and (2) a ternary Bi-Cu-V oxide system, which is a more challenging real system obtained from chemical experiments and is more noisy and uncertain. For each system, the input data points are mixtures of XRD patterns, associated with a composition graph identifying elemental compositions, and a constraint graph of data points, in which there is an edge between two data points if they share a constraint. 
For comparison with a recent simulation-based algorithm, we also analysed a powder system (Li-Sr-Al) from the article "A deep-learning technique for phase identification in multiphase inorganic compounds using synthetic XRD powder patterns".

- **Games** : Two illustrative games: Multi-MNIST-Sudoku (4x4 and 9x9)

These directories contain the relevant code and data, except the eBird data for joint species distribution modeling, which is provided separately due to large file sizes.



## Hardware Requirements

The code is mainly in Python using PyTorch or TensorFlow which is recommended to be run on a computer with GPUs (Otherwise, it takes too long to run it). 
For optimal performance, we recommend a computer with the following specs:

RAM: 32+ GB  
CPU: 8+ cores, 3.0+ GHz/core
GPU: NVIDIA Tesla V100 GPU with 16GB memory

The runtimes in the paper are generated using a computer with the recommended specs.

## Software Requirements

### OS Requirements

The code is tested on *Linux* operating systems. The developmental version of the package has been tested on the following systems:

Linux: Centos 7.8

### Dependency Requirements

The code is mainly in Python using PyTorch or TensorFlow. Dependencies that satisfy all applications include:

- Python 3.7
- PyTorch 0.4.1
- TensorFlow 1.8
- Torchvision 0.4.0
- Scikit-image
- Scikit-learn 0.19.2

(It takes less than 30 minutes to install these packages.)

Helper scripts also assume a Bourne or Bash shell, however they are short and can easily be converted to other platforms or entered manually.

## Usage

More information about each application and dataset is included in the readme.md files in each subdirectory. The scripts and instructions for replicating the results from each experiment in the publication are in these subdirectories:

- **Phase mapping with Al-Li-Fe dataset**: ./Crystal-Structure-Phase-Mapping/Al-Li-Fe-lib-159-I-compressed-new-231-solu/
- **Phase mapping with Bi-Cu-V dataset**: ./Crystal-Structure-Phase-Mapping/Bi-Cu-V-lib-100-I-compressed-new-307--solu/
- **Phase mapping with Li-Sr-Al powder system dataset**: ./Crystal-Structure-Phase-Mapping/Li-Sr-Al-powder-lib-38-50--fullQ-solu/

- **Multi-MNIST-Sudoku (4x4)**: ./Games/MNIST_Sudoku_4x4/
- **Multi-MNIST-Sudoku (9x9)**: ./Games/MNIST_Sudoku_9x9/
- **Multi-MNIST-Sudoku-Jupyter-Notebook-Demo**: ./Games/Multi-MNIST-Sudoku-jupyter-note-book.zip 
  (This demo takes around 30 minutes to converge with a normal GPU.)






