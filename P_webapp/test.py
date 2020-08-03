import streamlit as st
import os
import pandas as pd
from pandas import DataFrame
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)
# st.write("""
# # DASHBOARD
# A Simple Visualization Of Vital Features Of Patient
# """)



df = pd.read_csv("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\%s" %filename)
#df = DataFrame(df,columns=['Resp','MAP'])
st.line_chart(df)
st.area_chart(df)

df = pd.read_csv("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\DATA\data\%s" %filename)
df = DataFrame(df,columns=['O2Sat','HR'])
st.area_chart(df)
df = pd.read_csv("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\DATA\data\%s" %filename)
option = st.sidebar.selectbox(
    'Which number do you like best?',
     df['HR'])

'You selected:', option
