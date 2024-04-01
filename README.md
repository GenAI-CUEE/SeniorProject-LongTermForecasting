# GenAI-CUEE-SeniorProject-LongTermForecasting
SeniorProject LongTermForecasting is developed by Tanan Boonyasirikul

This repository describes the process to train and inferring Irradiance from CUEE Dataset by using a Deep neural network.

Categorized in 2 parts of DNNs: Transformer based Model ( Transformer, Autoformer, Informer)
                               
: Channel independent Model ( DLinear, NLinear, Linear and PatchTST)

For my result, select 7 features from CUEE dataset

- $I$ Irradiance (W/$m^2$)
- $I_clr$ Irradiance clear sky (W/$m^2$)
- $latt$ latitude for each sensor at CUEE department  
- $long$ longitude for each sensor at CUEE department  
- $day$ during which day was the data collected
- $month$ during which month was the data collected
- $hour$ collecting data at the hour of that day using Coordinated Universal Time (UTC)

Input format  : [ Number of Batch size, Sequence length, Number of features ]
Output format : [ Number of Batch size, Predict length, Number of features ]

| Files/Folders | Description |
|---------------|-------------|
|`pics/`     | Collecting result from univariate and multivariate time series forecasting and structure of model |
| `cuee_scripts/` | Execute scripts file to training and infer tasks for each model | 
|`exp` |  setting training and inference phase | 
| `run_longExp.py`| Training the model |
| `plot_longExp.py`| Inferring the model |
|`model` |  Structure for each model | 
|`data_provider` |  Dataset and DataLoader | 
|`layers` | Component in encoder decoder blocks, DataEmbedding | 


## Process

1. Create a `data` folder  and push "updated_measurement_Iclr_new.csv" into this folder.
2. Execute `cuee_scripts/Long-Exp/` choose the model for training.
3. Adjust hyper parameter and choose Sequence and Prediction length
   
        | Param         | Settings|
        |---------------|---------|
        | `model_name` | specify the model | 
        |`moving average` | Use the amount of lookback window to find moving average |
        | `batch_size` | the amount for data after split into mini batch | 
        | `label_len`     | start token length  | 
        | `seq_length`  | input sequence length  |
        | `target` | data features in forecasting problem | 
        | `pred_length` | prediction sequence length | 
        |`feature_type` |  "MS" and "S" for multivariate and univariate time series forecasting respectively|
        | `d_model` | dimension of model |
        | `embed_type` | 4 type of DataEmbedding (combination between value encoding, position encoding and temporal(time) encoding) |
        | `e_layers` | number of encoder layer |
        | `d_layers` | number of decoder layer |
        | `stride` | number of stride in data preprocessing for PatchTST model  | 
        | `patch_len` | sequence of data after patching |
        | `embed` | time embedding "timeF", original value of time "None" |
   
5. After that, Execute `cuee_scripts/Long-Exp/` choose the model for inferring.

  If choose mode="valids" : Result will contain in valids folder

  If choose mode="test"   : Result will contain in test folder 

7. For tuning parameter "d_model", Execute "plot_validation.py" to show the best parameters.
