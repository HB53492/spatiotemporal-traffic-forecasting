## SWA orchestrator
This experimental evaluation of an LSTM, GCN--LSTM, and STGCN for spatiotemporal traffic forecasting also served as an introduction to optimization through Stochastic Gradient Descent. As such, the notebook uses the SWA orchestrator found here: https://github.com/HB53492/SWA-Orchestrator-for-SGD. A note about SWA: given the training of the models, the `swa_lr_factor` should be larger, (something like 10.0 instead of 2.0) particulary for the STGCN, as 2 $\times$ 1e-6 was too small for any real exploration.

## Training
Tuning the training parameters for SGD was fickle, especially for the GCN--LSTM. Training and cross-model evaluation used the Huber loss function. Out of curiosity, I also compared STGCN performance using MSE as well. Training the STGCN with MSE required a halved learning rate (0.005 for MSE compared to 0.01 for Huber).

## Performance
STGCN optimized with Huber performed the best. The MSE variant had slightly better MSE than the Huber variant but had worse WAPE and MAE.
