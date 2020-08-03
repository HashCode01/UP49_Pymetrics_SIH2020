"""
File for extracting and saving different pieces of the data for use in various places. For example col names, continuous
col names, etc etc.
"""
from definitions import *
import sys
sys.path.insert(1, 'your_path\src')
from helper.functions import *
# Get data
# df = load_pickle(ROOT_DIR + '/data/test/preprecessed/formatted/df.pickle')
# df_raw = load_pickle(ROOT_DIR + '/data/test/preprocessed/from_raw/df.pickle')

# Cols
cts_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']

irregular_cols = [ 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                  'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
                  'Glucose', 'Magnesium',  'Potassium',
                  'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets']

derived_cols = ['ShockIndex', 'BUN/CR', 'SaO2/FiO2',
                'HepaticSOFA', 'MEWS', 'qSOFA', 'SOFA', 'SOFA_deterioration',
                'SepticShock', 'SIRS', 'SIRS_path']

other_cols = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'hospital']

# drop_cols = ['Lactate', 'WBC', 'Bilirubin_total', 'Alkalinephos', 'PaCO2', 'Hct', 'pH',
#              'TroponinI', 'PTT', 'Hgb', 'O2Sat']
drop_cols = ['Lactate', 'Bilirubin_total', 'Alkalinephos', 'PaCO2', 'pH', 'Hct',
             'TroponinI']
