## Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation
This repository contains a Python implementation of the methods described in the paper "Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation." 

### Abstract
We introduce a technique which can be used to manipulate the noise present in financial data in order to better estimate a covariance matrix. The technique, which we refer to as squeezing, parameterizes statistical distributional alignment so that we can vectorize co-movement noise. Squeezing underpins a novel approach to portfolio optimization in which the covariance matrix may be determined on an objective-specific basis. Our model-free approach more fully explores the eigenspace of the estimated matrix and is applicable across the dimensionality range of portfolio size and concentration. Squeezing is shown to outperform popular techniques used to treat noise in financial covariance matrices.

#### Installation
To use this repository, you will need to have Python 3.6 or higher installed. Additionally, it is recommended to create a virtual environment. You can install the required packages using pip:
```bash
cd src
pip install -r requirements.txt
```

#### Preparing the Dataset
The expected input format is a CSV file where: Each row represents a total return index at different time period. Each column represents a different asset. See an example for the format of the data. 
Place your prepared dataset in the `data/prcs.csv`.
```csv
Date,SPX,RTY,M1EA,EM,XAU,SPGSCI,LF98TRUU,LBUSTRUU,FNERTR
1988-01-29,data1,data2,data3,data4,data5,data6,data7,data8,data9 
```

#### Running the Code
To execute the code, use the following command:
```bash
cd src
python run_mvo_iq.py
```

#### Results
After running the code, results will be saved in the results/ directory. You can find:
- Estimated covariance matrices under `results/prcs/covs`.
- Summary statistics and reports under `results/prcs/portfolio`.
- Plots comparing the performance of different estimation methods under `results/prcs`.




