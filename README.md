# IQN_HEP

### To Train an IQN model,

`python run_training.py --T <target>`

with the target T being an option from [RecoDatapT, RecoDataeta, RecoDataphi, RecoDatam].
For example, 

### `python run_training.py --T RecoDatapT`

This generates a trained pytorch model parameter dictionary in `trained_models/`, as well as compare with the `RecoData` target sample that you're estimating, and produce a comparison plot in `images/`

