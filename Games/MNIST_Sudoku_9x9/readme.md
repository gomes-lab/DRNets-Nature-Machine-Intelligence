# Multi-MNIST-Sudoku (9x9) code and dataset

## Requirements

- Python 3.6
- Pytorch 0.4.1
- torchvision 
- skimage
- scikit-learn

## MNIST/EMNIST Dataset

Due to the large size of the original EMNIST dataset, we only used a subset of it to generate the Sudokus with letters. To better organize our data, we stored handwritten digits and handwritten letters in two zipped dictionaries: "selected_digits_offset2.npy.zip" and "selected_uppers_offset2.npy.zip".

These two dictionary files should be unzipped into the current directory.

## Usage

For the experimental setting where the overlapping Sudoku puzzles contain different digits/letters (1-9 and A-I):

```
#  Generate Sudoku data where each digit is sampled from the MNIST/EMNIST dataset (takes about 10-20 mins)
python3 gen_9by9_data.py 10000 123 

# Solve mixed Sudokus. On a machine with an NVIDIA Tesla V100 GPU, it takes 
# about 7 hours to converge. Manually terminate the training when it has 
# converged to the desired criteria (otherwise it will run the maximum number 
# of epochs).
python3 sep_9by9.py 

```


For the ablation study removing the generative decoder (w/o cGAN):

```
python3 sep_9by9_nogan.py 

```

For the ablation study removing the reasoning module (w/o reasoning module):

```
python3 sep_9by9_noreason.py

```

For the downscale study:

```
python3 sep_9by9_downscale.py

```

In all experiments, the results are printed to stdout.
