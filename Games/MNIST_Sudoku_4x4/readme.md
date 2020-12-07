# Multi-MNIST-Sudoku (4x4) code and dataset

## Requirements

- Python 3.6
- Pytorch 0.4.1
- torchvision 
- skimage
- scikit-learn
 
## Usage

For the experimental setting where the overlapping Sudoku puzzles contain different digits (1-4 and 5-8):

```
#  Generate Sudoku data where each digit is sampled from the MNIST test set (takes 
# about 2-3 mins)
bash generate_data.sh

# (Optional) train a conditional GAN model from scratch. A pretrained model is 
# provided at ./models/G-180.pt , if you prefer to skip this step.
python3 cgan.py

# Solve mixed Sudokus. On a machine with an NVIDIA Tesla V100 GPU, it takes 
# about 30mins to converge. Manually terminate the training when it has 
# converged to the desired criteria (otherwise it will run the maximum number 
# of epochs).
python3 sep.py
```

For the experimental setting where the overlapping Sudoku puzzles can share digits:

```
# Generate Sudoku data where <x> and <y> define the range of values for the 
# second puzzle (the first puzzle is assumed to still use the digits 1-4). 
# x should be between 1 and 5, and y=x+3.
python3 gen_data_common_digits.py --st1 1 --ed1 4 --st2 <x> --ed2 <y>

# Solve the mixed Sudokus, where z is the number of common digits shared by 
# both puzzles, i.e. <z>=5-x.
python3 sep.py --ocase <z>
```

In both experiments, the results are printed to stdout.
