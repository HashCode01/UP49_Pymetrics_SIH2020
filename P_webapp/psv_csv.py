import pandas as pd
import csv

df=pd.read_csv("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\p1.psv",sep='|')
df.to_csv("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\p1.csv")

# import pandas as pd
# import pickle

# df=pd.read_pickle("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\probas.pickle")
# # df
# pickle_in = open("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\df.pickle","rb")
# example_dict = pickle.load(pickle_in)
