## SWA orchestrator
This experimental evaluation of an LSTM, GCN--LSTM, and STGCN for spatiotemporal traffic forecasting also served as an introduction to optimization through Stochastic Gradient Descent. As such, the notebook uses the SWA orchestrator found here: https://github.com/HB53492/SWA-Orchestrator-for-SGD. Furthermore, training and evaluation used the Huber Loss function with varying degrees of success.

## Training
Tuning the training parameters for SGD was fickle, especially for the GCN--LSTM.

## Performance
STGCN performed the best, but large outliers were missed, likely do to the Huber loss function.
