# IQN_HEP

### To Train a pytorch IQN model,

`python run_training.py --T <target>`

with the target T being an option from [RecoDatapT, RecoDataeta, RecoDataphi, RecoDatam].
For example, 

### `python run_training.py --T RecoDatapT`

This uses the "gen" and "reco" level jets (8 columns + $\tau$), `Data.csv`, which is composed of 100,000 examples, and generates a trained pytorch model parameter dictionary in `trained_models/`. It also compares with the `RecoData` target sample that you're estimating, and produces a comparison histogram plot in `images/`.

