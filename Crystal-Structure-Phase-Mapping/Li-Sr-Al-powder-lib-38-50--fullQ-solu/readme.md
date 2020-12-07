## Training step

```
bash runTrain.sh 

(Don't let it run to the end. You can stop it just after 10k iterations or you see that the reconstruction error converges.)
(Use bash TB.sh to launch the tensorboard and view the training curves.)
```

## Post-processing step: cut-off insignificant weights (<1%)
```
python refine.py

```

## Fine-tuning step:
```
python train.py refine con

(you can stop it when the reconstruction error converges, e.g., 4k iterations)

```

## Testing step
```
# For reconstruction error statistics
python test.py refine recon

# Visualize the phase concentration map on the ternary system:
python test.py refine ternary

# Generate a solution file (solu.txt)
python test.py refine solu 
```
	
