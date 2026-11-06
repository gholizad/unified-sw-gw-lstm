# unified-sw-gw-lstm

## Model implementation and citation

The groundwater (GW) and streamflow (SW) models use the native `CudaLSTM` implementation from [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology), configured as `model: cudalstm`. This repository provides the custom dataset loaders and experiment configurations. The original runs used NeuralHydrology **1.13.0**.

Please cite the underlying NeuralHydrology software as:

Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022). *NeuralHydrology — A Python library for Deep Learning research in hydrology*. Journal of Open Source Software, 7(71), 4050. [https://doi.org/10.21105/joss.04050](https://doi.org/10.21105/joss.04050).

