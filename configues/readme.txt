this configue files has been made for training parameters. each 'key' in each jason
file represents the hyper parameters we want to use for tuning our model.

we have two kind of configues. first one is cnn and it contains 'cnn' preprocessing and
training parameters. second one contains 'mlp' configues for state feature extraction.


read the 'readme.txt' in each folder to see how each parameters works.



the tree structure of this directory described as below:


├── cnn
│   ├── grid
│   │   ├── domain_and_parameters.json
│   │   ├── domain_randomization.json
│   │   └── single.json
│   ├── optuna
│   │   ├── domain_and_parameters.json
│   │   ├── optional
│   │   │   ├── cnn_lstm.json
│   │   │   ├── mobilenet.json
│   │   │   └── small_large.json
│   │   └── parameters_tuning.json
│   └── readme.txt
├── mlp
│   ├── grid
│   │   ├── domain_and_parameters.json
│   │   ├── domain_randomization.json
│   │   └── single.json
│   ├── optuna
│   │   ├── domain_and_parameters.json
│   │   └── parameters_tuning.json
│   └── readme.txt
└── readme.txt
