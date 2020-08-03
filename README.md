GEHC-SepsisChallengeSIH 2020
==============================

The repository contains the code for our submission (Team Name: Pymetrics) to the GEHC-SepsisChallengeSIH 2020. The code has not been edited since the final submission and as such is a little untidy, we will be working on making the codebase more userfriendly in future, please excuse the current mess!

**Docter's Dashboard:**

![image](P_webapp/src/Pymetrics_logo_t-01.png =250x250)

Getting Started
---------------
**Note** :Once the repo has been cloned locally, setup a python environment and add the path for the folder according to your directory.

you must download the official PhysioNet training data. This consists of two folders of .psv files. These folders should be placed in data/raw/training_{A, B}. 

To setup the data run ``src/data/make_dataframe.py``.

To Generate Model run the src/models/model.ipynb in any respective notebook . We used Google Colab. No Data files have been added here.

# From
# Setup paths
DATA_DIR = ROOT_DIR + '/data/test'
MODELS_DIR = ROOT_DIR + '/models/test'

# To
# Setup paths
DATA_DIR = ROOT_DIR + '/data'
MODELS_DIR = ROOT_DIR + '/models'



Project Organization
--------------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data   
    │   ├── preprocessed   <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── Reference          <- Papers that may be useful for the project.
    │
    ├── models             <- Script for training , testing , validation using Bidirectional LSTM
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── reports           
    │   ├── Presentation         <- SIH 2020 Finale Presentation.
    │	└── Technical paper      <- Technical paper based on solution design .
    │
    ├── src                      <- Source code for use in this project.
    │   ├── __init__.py          <- Makes src a Python module
    │   │
    │   ├── data                 <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features             <- Scripts to turn raw data into features for modeling
    │   │   ├── signatures 
    │   │   |    └── compute_signatures.py 
    │   │   |    └── signature_functions.py 
    │   │   |    └── transformers.py 
    │   │   └── build_features.py   
    │   │
    │   ├── models              <- Scripts to train models and then use trained models to make
    │   │   │                       predictions
    │   │   └──model.ipynb
    │   │
    │   └──helper               <- Generic functions used everywhere, such as load_pickle
    │   
    │        
    │   
    ├── P_webapp   
    │   ├── src                 <- Contains all the resources like pngs and screenshots.
    │   ├── data                <- All the Finally imputed psv files after train.
    │   ├── webapp_doc.py       <- Source code of the webapp(For Docters).
    │ 	└── webapp_patient.py   <- Source code of the webapp(For Patients).
    │ 
    │
    └── definitions.py          <- File to load useful global info such as ROOT_DIR 


--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
