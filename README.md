## Setup
* clone this repo
* get python + Tensorflow(>=2.1)
* clone [mantflow](https://github.com/tum-pbs/mantaflow)
* add the content of NeuralTurbulence/plugin into mantaflow/source/plugin and to the CMakeLists.txt
* build manta

## Usage
To train a network, just update the parameters and architecture in `manta_trainGenerator.py`, then run
`manta manta_trainGenerator.py`
If you want to evaluate a network or get validation errors during training, you first need to generate a suitable dataset with the option `writeData=True`. Afterwards, 
`python runmodel.py <models> <params>` 
provides a varity of options to compute different error measures and visualize the outputs.
