The core function of the PBCT algorithm is included in the file utils/PBCT.py. Given the labeled and unlabeled training data as well as the test data, it triggers the training of the complete-view model and parital-view models, save the model parameters in the desired paths, and return the test error measured using RMSE. An example for utilizing the PBCT algorithm is provided in the __main__ section of this file.

This repository is a reproduction work of the PBCT algorithm on the field data, which can be found in [dataset](https://github.com/TengMichael/battery-charging-data-of-on-road-electric-vehicles). The main changes are made in the folder: /utils.

