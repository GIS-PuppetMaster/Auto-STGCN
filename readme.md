# Auto-STGCN
An automated system for STGCN model development.<br>
Code for paper 'Auto-STGCN: Autonomous Spatial-Temporal Graph Convolutional Network Search Based on Reinforcement Learning and Existing Research Results'.<br>

## 1. Auto-STGCN Algorithm: Searching for the optimal STGCN model
### Related Files
>Auto_STGCN.py --- run Auto-STGCN algorithm<br>
>Model.py --- build STGCN model according to code<br>
>Env.py --- read dataset, record the state-action-reward information in Auto-STGCN algorithm<br>
>ExperimentDataLogger.py --- output the log information of Auto-STGCN algorithm<br>
>/Log --- log files<br>
>/utils --- auxiliary files<br>
>/data --- datasets<br>
>/Config --- default configurations<br>

### Inputs Details
* Dataset name, Dataset partition ratio (validation set, test set, training set), Input sequence length, Output sequence length,<br>
* Timemax, Epoch size of each candidate model,<br>
* Initial epsilon, Epsilon decay step, Epsilon decay Ratio, Gamma of Qlearning, Learning rate of Qlearning, Episodes of Qlearning<br>

### Outputs Details
* Code and performance scores of the Optimal STGCN searched by Auto-STGCN<br>
* Log info of Auto-STGCN<br>

### Commands
* `python Auto_STGCN.py --data "PEMS03"`<br>
* `python Auto_STGCN.py --data "PEMS03" --gamma 0.1`<br>

## 2. Auto-STGCN Algorithm: Training the optimal STGCN model
### Related Files
>TestBestGNN.py --- train the optimal STGCN model searched by Auto-STGCN algorithm<br>
>Model.py --- build STGCN model according to code<br>
>/Log --- log files<br>
>/utils --- auxiliary files<br>
>/data --- datasets<br>
>/Config --- default configurations<br>

### Inputs Details
* Optimal STGCN code, Dataset name, Dataset partition ratio (validation set, test set, training set), Input sequence length, Output sequence length,<br>
* Model training epochs, Model training times,<br>
* Load model weight = None<br>

### Outputs Details
* Performance scores (Mean + variance: MAE, MAPE, RMSE, Time) of the Optimal STGCN model<br>
* Log info of the model training<br>

### Commands
* `python TestBestSTGNN.py --model "./Config/qlearning_2.json" --data "PEMS03"`<br>
* `python TestBestSTGNN.py --model "./Config/qlearning_2.json" --data "PEMS03" --gamma 0.1`<br>

## 3. Auto-STGCN Algorithm: Loading the optimal STGCN model
### Related Files
>TestBestGNN.py --- test the performance of optimal STGCN model searched by Auto-STGCN algorithm<br>
>Model.py --- build STGCN model according to code<br>
>/Log --- log files<br>
>/utils --- auxiliary files<br>
>/data --- datasets<br>
>/Config --- default configurations<br>

### Inputs Details
* Dataset name, test number, Load model weight = Model loading path <br>

### Outputs Details
* Performance scores (Mean + variance: MAE, MAPE, RMSE, Time) of the Optimal STGCN model on test set<br>

### Commands
* `python TestBestGNN.py --data "PEMS03" --load "./Log/PEMS03_experiment2_qlearning2_test/GNN/best_GNN_model.params" --times 1`<br>
