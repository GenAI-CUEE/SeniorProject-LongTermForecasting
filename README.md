# GenAI-CUEE-SeniorProject-LongTermForecasting
SeniorProject LongTermForecasting is developed by Tanan Boonyasirikul

This repository describes the process of training and inferring Irradiance from CUEE Dataset by using a Deep neural network.Categorized in 2 parts of DNNs

__1. Based line model ( Regression Long-Short Term Memory (RLSTM) )__

   This assumption model applied with recurrent neural network (RNN) model which the concept is keeping or removing previous sequence data.
                
![Structure RLSTM](https://github.com/GenAI-CUEE/GenAI-CUEE-SeniorProject-LongTermForecasting/assets/145090574/c446ff7f-655a-441a-94eb-a0200684ee2e)

__2. Channel independent Model ( DLinear, NLinear, Linear )__
   
   This assumption model is channel features are independent and not correlate between other channel in time series forecasting tasks by represent vanila linear model to show that time series features are not correlated.
   
![Structure Linear](https://github.com/GenAI-CUEE/GenAI-CUEE-SeniorProject-LongTermForecasting/assets/145090574/1e675eda-3bea-470d-a5c5-17b9543bf6ae)

   __DLinear :__ using series decomposition and split in term of trend and seasonality    component then aggregate together.
   
![Structure DLinear](https://github.com/GenAI-CUEE/GenAI-CUEE-SeniorProject-LongTermForecasting/assets/145090574/f163324d-5b48-495f-bde9-870ce7c46e11)
   
   __NLinear :__ using when there is a distribution shifting problem in dataset, subtract the last value sequence then pass through a linear model then add the subtracted part into the model.

![Structure NLinear](https://github.com/GenAI-CUEE/GenAI-CUEE-SeniorProject-LongTermForecasting/assets/145090574/a8eab30c-c8ee-4ae3-8384-6dd385d4ddab)

__3. PatchTST__
   
   This assumption model using channel independent concept and adding patching teqnique to convert sequence length in term of patch number and each patch has the same length. Patching number and patch length is the hyper parameter for this model

![Structure PatchTST](https://github.com/GenAI-CUEE/GenAI-CUEE-SeniorProject-LongTermForecasting/assets/145090574/9f12ac4e-4437-45c2-91b9-e4cee857910a)

__4. Transformer-Based ( Transformer, Autoformer and Informer )__

   This assumption model is using self-attention component that show the correlation of each features are correlate with the equation $Self-Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ The another main concept is encoder-deconder component and each model using different type of self-attention 

   __Transformer :__ vanila transformer using encoder decoder models with self-attention.
   
![Enc_Dec](https://github.com/GenAI-CUEE/GenAI-CUEE-SeniorProject-LongTermForecasting/assets/145090574/7c9d3801-760a-462c-924e-de099a0c7f2c)


   __Autoformer :__ applied with auto-correlation between channels.

![Autocorrelation Block in Autoformer](https://github.com/GenAI-CUEE/GenAI-CUEE-SeniorProject-LongTermForecasting/assets/145090574/cc51861d-92bc-4c3d-8c72-b1f8e63d7b4b)

   __Informer :__ adjusted self-attention that use only high score from self attention to represent correlation.
   

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   

For my result, select 7 features from CUEE dataset ( Not use Zenith angle, Airmass coefficient and Clear-sky index )

- $I$ Irradiance (W/m^2)
- $I_{clr}$ Irradiance clear sky (W/m^2)
- $latt$ latitude for each sensor at CUEE department  
- $long$ longitude for each sensor at CUEE department  
- $day$ during which day was the data collected
- $month$ during which month was the data collected
- $hour$ collecting data at the hour of that day using Coordinated Universal Time (UTC)

Prediction with unnormalize data.
![Predicting Result For Multifeatures Time Series Forecasting ( Normalize value )](https://github.com/GenAI-CUEE/GenAI-CUEE-SeniorProject-LongTermForecasting/assets/145090574/2ccb4760-5f9e-4cc3-ba7d-9ab17293a2d4)

Prediction with normalize data
![Predicting Result for Univariate Time Series Forecasting ( Real Irradiance value )](https://github.com/GenAI-CUEE/GenAI-CUEE-SeniorProject-LongTermForecasting/assets/145090574/508f088c-3067-4462-832c-df463f6a48fc)


Input format  : [ Number of Batch size, Sequence length, Number of features ]

Output format : [ Number of Batch size, Predict length, Number of features ]

| Files/Folders | Description |
|---------------|-------------|
|`pics/`     | Collecting results from univariate and multivariate time series forecasting and structure of model |
| `cuee_scripts/` | Execute scripts file to training and infer tasks for each model | 
|`exp` |  setting training and inference phase | 
| `run_longExp.py`| Training the model |
| `plot_longExp.py`| Inferring the model |
|`model` |  Structure for each model | 
|`data_provider` |  Dataset and DataLoader | 
|`layers` | Component in encoder-decoder blocks, DataEmbedding | 


## Process

1. Create a `data` folder  and push "updated_measurement_Iclr_new.csv" into this folder.
2. Execute `cuee_scripts/Long-Exp/` and choose the model for training.
3. Adjust the hyperparameter and choose the Sequence and Prediction length
   
| Param         | Settings|
|---------------|---------|
| `model_name` | specify the model | 
|`moving average` | Use the amount of lookback window to find the moving average |
| `batch_size` | the amount for data after split into mini batch | 
| `label_len`     | start token length  | 
| `seq_length`  | input sequence length  |
| `target` | data features in forecasting problem | 
| `pred_length` | prediction sequence length | 
|`feature_type` |  "MS" and "S" for multivariate and univariate time series forecasting respectively|
| `d_model` | dimension of model |
| `embed_type` | 4 types of DataEmbedding (combination between value encoding, position encoding and temporal(time) encoding) |
| `e_layers` | number of encoder layer |
| `d_layers` | number of decoder layer |
| `stride` | number of strides in data preprocessing for PatchTST model  | 
| `patch_len` | sequence of data after patching |
| `embed` | time embedding "timeF", the original value of time "None" |
   
5. After that, Execute `cuee_scripts/Long-Exp/` choose the model for inferring.

          If choose mode="valids" : Result will be contained in valid folder

          If choose mode="test"   : Result will contain in test folder 

7. For tuning parameter "d_model", Execute "plot_validation.py" to show the best parameters.

# Citation
@article{2020RNN,
   title={Fundamentals of Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) network},
   volume={404},
   ISSN={0167-2789},
   url={http://dx.doi.org/10.1016/j.physd.2019.132306},
   DOI={10.1016/j.physd.2019.132306},
   journal={Physica D: Nonlinear Phenomena},
   publisher={Elsevier BV},
   author={Sherstinsky, Alex},
   year={2020},
   month=mar, pages={132306} }

@article{1997LSTM,
author = {Hochreiter, Sepp and Schmidhuber, J\"{u}rgen},
title = {Long Short-Term Memory},
year = {1997},
issue_date = {November 15, 1997},
publisher = {MIT Press},
address = {Cambridge, MA, USA},
volume = {9},
number = {8},
issn = {0899-7667},
url = {https://doi.org/10.1162/neco.1997.9.8.1735},
doi = {10.1162/neco.1997.9.8.1735}, 
journal = {Neural Comput.},
month = {nov},
pages = {1735–1780},
numpages = {46}
}


@misc{staudemeyer2019LSTM,
      title={Understanding LSTM -- a tutorial into Long Short-Term Memory Recurrent Neural Networks}, 
      author={Ralf C. Staudemeyer and Eric Rothstein Morris},
      year={2019},
      eprint={1909.09586},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}


@inproceedings{NIPS2017_Attention,
 author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, \L ukasz and Polosukhin, Illia},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {I. Guyon and U. Von Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Attention is All you Need},
 url = {https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf},
 volume = {30},
 year = {2017}
}

@inproceedings{wu2021_Autoformer,
  title={Autoformer: Decomposition Transformers with {Auto-Correlation} for Long-Term Series Forecasting},
  author={Haixu Wu and Jiehui Xu and Jianmin Wang and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}


@book{forecasting_book,
title = "Forecasting: Principles and Practice",
author = "Rob Hyndman and G. Athanasopoulos",
year = "2021",
language = "English",
publisher = "OTexts",
address = "Australia",
edition = "3rd",
}


@inproceedings{Zeng2022AreTE,
  title={Are Transformers Effective for Time Series Forecasting?},
  author={Ailing Zeng and Muxi Chen and Lei Zhang and Qiang Xu},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}


@inproceedings{haoyietal-informer-2021,
  author    = {Haoyi Zhou and
               Shanghang Zhang and
               Jieqi Peng and
               Shuai Zhang and
               Jianxin Li and
               Hui Xiong and
               Wancai Zhang},
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference},
  volume    = {35},
  number    = {12},
  pages     = {11106--11115},
  publisher = {{AAAI} Press},
  year      = {2021},
}

@misc{triebe2021neuralprophet,
      title={NeuralProphet: Explainable Forecasting at Scale}, 
      author={Oskar Triebe and Hansika Hewamalage and Polina Pilyugina and Nikolay Laptev and Christoph Bergmeir and Ram Rajagopal},
      year={2021},
      eprint={2111.15397},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
