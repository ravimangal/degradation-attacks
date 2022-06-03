# Degradation Attacks on Certifiably Robust Neural Networks
This repository contains code for the paper "Degradation Attacks on Certifiably Robust Neural Networks". 

## What's happening to the certifiably robust neural networks?
Certifiably robust neural networks protect against adversarial examples by employing provable run-time defenses that check if the model is locally robust at the input under evaluation. We show through examples and experiments that even complete defenses are inherently over-cautious. 
Specifically, they flag inputs for which local robustness checks fail, but yet that are not adversarial; 
i.e., they are classified consistently with all valid inputs within a distance of ε. 
As a result, while a norm-bounded adversary cannot change the classification of an input, it can use norm-bounded changes to degrade the utility of certifiably robust networks by forcing them to reject otherwise correctly classifiable inputs. 

## How to reproduce the results in the paper?

### Get started 

1. Clone the repository via
`git clone https://github.com/ravimangal/degradation-attacks.git`

2. Install from source via
```
conda create --name degrade
conda activate degrade
conda install -c conda-forge tensorflow 
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install cachable scriptify tensorflow_datasets matplotlib pandas seaborn foolbox
```

### Lower Bound experiments
In these experiments, we compute lower bounds on the efficacy of the degradation attacks using our attack algorithms. 
File `experiments/dos/script_attack_gloro.py` calculate the attack results on GloRo models, `experiments/dos/script_attack_rs.py` calculate the results on Randomized Smoothing models, and `experiments/dos/script_attack_kw.py` on KW models. 

### Upper Bound experiments
In these experiments, we compute upper bounds on the efficacy of degradation attacks, i.e., upper bounds on the false positive rates. 
For GloRo, we train our own models (`trainGloro.py`) and calculate their certified radii (`printRadius.py`).  For Randomized Smoothing models, we use the data generated by Cohen et al. available at `https://github.com/locuslab/smoothing/tree/master/data/certify`. 
For generating the plots in the paper, we use the file `analyze.py`.

#### experiments/trainGloro.py

This file trains the gloro models. Hyperprameters for training and evaluating the model should be provided in the command line as follows.
```
python experiments/trainGloro.py --superrobust='Y' --dataset="mnist" --architecture="minmax_cnn_2C2F" --epsilon=0.3 --epsilon_train=0.3 --epsilon_schedule='fixed' --loss='sparse_trades.2.0' --augmentation=None --epochs=500 --batch_size=128 --optimizer='adam' --lr=1e-3 --lr_schedule='decay_to_0.000001' --trades_schedule='linear_from_0.1'

python experiments/trainGloro.py --superrobust='N' --dataset="mnist" --architecture="minmax_cnn_2C2F" --epsilon=0.3 --epsilon_train=0.3 --epsilon_schedule='fixed' --loss='sparse_trades.0.1' --augmentation=None --epochs=500 --batch_size=128 --optimizer='adam' --lr=1e-3 --lr_schedule='decay_to_0.000001' --trades_schedule='linear_from_0.1'

python experiments/trainGloro.py --superrobust='Y' --dataset="mnist" --architecture="minmax_cnn_4C3F" --epsilon=1.58 --epsilon_train=1.74 --epsilon_schedule='logarithmic' --loss='sparse_trades.1.5' --augmentation=None --epochs=500 --batch_size=128 --optimizer='adam' --lr=1e-3 --lr_schedule='decay_to_0.000005' --trades_schedule='fixed'

python experiments/trainGloro.py --superrobust='N' --dataset="mnist" --architecture="minmax_cnn_4C3F" --epsilon=1.58 --epsilon_train=1.74 --epsilon_schedule='logarithmic' --loss='sparse_trades.1.5' --augmentation=None --epochs=500 --batch_size=128 --optimizer='adam' --lr=1e-3 --lr_schedule='decay_to_0.000005' --trades_schedule='fixed'

python experiments/trainGloro.py --superrobust='Y' --dataset="cifar10" --architecture="minmax_cnn_6C2F" --epsilon=0.141 --epsilon_train=0.141 --epsilon_schedule='logarithmic' --loss='sparse_trades.1.2' --augmentation='cifar' --epochs=800 --batch_size=512 --optimizer='adam' --lr=1e-3 --lr_schedule='decay_to_0.000001' --trades_schedule='fixed'

python experiments/trainGloro.py --superrobust='N' --dataset="cifar10" --architecture="minmax_cnn_6C2F" --epsilon=0.141 --epsilon_train=0.141 --epsilon_schedule='logarithmic' --loss='sparse_trades.1.2' --augmentation='cifar' --epochs=800 --batch_size=512 --optimizer='adam' --lr=1e-3 --lr_schedule='decay_to_0.000001' --trades_schedule='fixed'
```

#### experiments/upper_bounds/printRadius.py
Print out the correct label, model predicted label, certified robust radius and correctness for each point of test data in the given dataset, using the given gloro models. 
The input model could either be the model generated by file `trainGloro.py` or an existing model in the `models` directory. 
An example command to run this file is as follows.

```
python experiments/upper_bounds/printRadius.py --dataset="cifar10" --batch_size=128 --model=cifar10_0.14_N
```


#### experiments/upper_bounds/analyze.py
Generate the plots in the paper. 
The data could from both gloro model (by file `printRadius.py`) and randomized smoothing model. 
The generated data for both models are stored in the `data` folder.

```
python experiments/upper_bounds/analyze.py
```
