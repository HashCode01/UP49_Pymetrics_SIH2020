import streamlit as st
import pandas as pd
import numpy as np
import time
from pandas import DataFrame
import os
st.write("""
# DOCTER'S DASHBOARD

A Simple Visualization Of Vital Features Of Patient


""")

# import time
# import numpy as np

# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()
# last_rows = np.random.randn(1, 1)
# chart = st.line_chart(last_rows)

# for i in range(1, 101):
#     new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#     status_text.text("%i%% Complete" % i)
#     chart.add_rows(new_rows)
#     progress_bar.progress(i)
#     last_rows = new_rows
#     time.sleep(0.05)

# progress_bar.empty()



# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
# st.button("Re-run")
# def file_selector(folder_path='.'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)

# filename = file_selector()
# st.write('You selected `%s`' % filename)

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a respective patient file :', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write(""" ## Patient selected : `%s`""" % filename[len(filename)-11:len(filename)-4])

from PIL import Image
image = Image.open('G:\SIH_2020\SIH_2020(Sp)\src\Pymetrics_logo_t-01.png')
st.sidebar.image(image,use_column_width=True)
df = pd.read_csv("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\data.csv")
option = st.sidebar.selectbox(
    'Enter The Patient No.',df['HR'])

'Patient selected:', option

df = pd.read_csv(option)
df = DataFrame(df,columns=['Resp','MAP'])
st.markdown("<h3 style='text-align: center; color: teal;'>RESP - MAP</h3>", unsafe_allow_html=True)
st.line_chart(df)

st.markdown("<h3 style='text-align: center; color: teal;'>RESP - MAP</h3>", unsafe_allow_html=True)
st.area_chart(df)

df = pd.read_csv("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\p000001.csv")
df = DataFrame(df,columns=['O2Sat','HR'])
st.markdown("<h3 style='text-align: center; color: teal;'>O2Sat - HR</h3>", unsafe_allow_html=True)
st.line_chart(df)
# df = pd.read_csv("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\p000001.csv")
df = pd.read_csv("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\p000001.csv")
df = DataFrame(df,columns=sorted(list(df.columns)))

columns = st.multiselect(
    label='What column to you want to display', options=df.columns)

st.line_chart(df[columns])
df = pd.read_csv("G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\p000001.csv")
if st.checkbox('Show dataframe'):
    st.dataframe(pd.DataFrame(df.transpose()))

from PIL import Image
image = Image.open('G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\GE_lg-01.png')
st.sidebar.image(image,use_column_width=True)

from PIL import Image
image = Image.open('G:\SIH_2020\SIH_2020(Sp)\StreamlitApp\SIH2020_lg-01.png')
st.sidebar.image(image, caption='#SMART INDIA HACKATHON 2020\n',
        use_column_width=True)

n=60
if n>50 :   
       my_slot1 = st.empty()
       st.markdown("<h3 style='text-align: left; color: red;'>Risky</h3>", unsafe_allow_html=True)
progress_bar = st.progress(0)
status_text = st.empty()
for i in range(n):

    progress_bar.progress(i + 1)
    status_text.text(
        'Calculating...')

                
    time.sleep(0.015)
status_text.text('Risky')
st.markdown("<h3 style='text-align: left; color: red;'>Risky</h3>", unsafe_allow_html=True)
status_text.text('Risky')


n=40
if n>50 :
       st.markdown("<h3 style='text-align: left; color: red;'>Risky</h3>", unsafe_allow_html=True)
if n<50 :
          st.markdown("<h3 style='text-align: left; color: Green;'>Healthy</h3>", unsafe_allow_html=True)          
status_text = st.empty()
progress_bar1 = st.progress(0)
for i in range(n):

    progress_bar1.progress(i + 1)
                
    time.sleep(0.015)
