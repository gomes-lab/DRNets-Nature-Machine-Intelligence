## Training step

```
# Don't let it run to the end. You can stop it just after 20k~30k 
# iterations or you see that the reconstruction error converges.
# Use bash TB.sh to launch tensorboard and view the training curves.
bash runTrain.sh 
```

## Post-processing step: cut-off insignificant weights (<1%)

```
python refine.py
```

## Fine-tuning step:

```
# You can stop it when the reconstruction error converges
python train.py refine con
```

## Testing step
```
# For reconstruction error statistics
python test.py refine recon

# Visualize the phase concentration map on the ternary system:
python test.py refine ternary

# Generate a solution file (solu.txt)
python test.py refine solu 


# Run the downscale version of DRNets, which solves one XRD pattern at a time
python train_downscale.py
```
