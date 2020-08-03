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


#Image db

from PIL import Image
image = Image.open('src\P_webapp\src\db_d (2).jpg')
st.image(image,use_column_width=True)

#Folder Path Selctor

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path+"/data")
    selected_filename = st.selectbox('Select a respective patient file :', filenames)
    return os.path.join(folder_path+"/data", selected_filename)

filename = file_selector()
st.write(""" ## Patient selected : `%s`""" % filename[len(filename)-11:len(filename)-4])

#Sepsis Label
df = pd.read_csv("your_path_folder\%s" %filename)
n=int(df.iloc[-1]['SepsisLabel'])

if n>0:
          st.markdown("<h2 style='text-align: left; color: red;'>Sepsis Positive</h2>", unsafe_allow_html=True)
          st.markdown("<h4 style='text-align: left; color: grey;'>Notification to the Docter as well as patient (initially we will consider email/sms)</h4>", unsafe_allow_html=True)
else:
          st.markdown("<h2 style='text-align: left; color: green;'>Sepsis Negetive</h2>", unsafe_allow_html=True)

#Image on Sidebar

from PIL import Image
image = Image.open('src\P_webapp\src\Pymetrics_logo_t-01.png')
st.sidebar.image(image,use_column_width=True)

#In DevMode
df = pd.read_csv("your_path_folder\data.csv")
option = st.sidebar.selectbox(
    'Enter The Patient No.',df['HR'])

#Image On Sidebar

from PIL import Image
image = Image.open('src\P_webapp\src\GE_lg-01.png')
st.sidebar.image(image,use_column_width=True)

from PIL import Image
image = Image.open('src\P_webapp\src\SIH2020_lg-01.png')
st.sidebar.image(image, caption='#SMART INDIA HACKATHON 2020\n',
        use_column_width=True)

# Line Chart 

df = pd.read_csv("your_path_folder\%s" %filename)
df = DataFrame(df,columns=['Temp'])
st.markdown("<h3 style='text-align: left; color: teal;'>Tempurature</h3>", unsafe_allow_html=True)
st.line_chart(df)

df = pd.read_csv("your_path_folder\%s" %filename)
df = DataFrame(df,columns=['HR'])
st.markdown("<h3 style='text-align: left; color: teal;'>Heart Rate</h3>", unsafe_allow_html=True)
st.line_chart(df)

df = pd.read_csv("your_path_folder\%s" %filename)
df = DataFrame(df,columns=['Resp'])
st.markdown("<h3 style='text-align: left; color: teal;'>Respiration</h3>", unsafe_allow_html=True)
st.line_chart(df)

#Show Data

df = pd.read_csv("your_path_folder\%s" %filename)
if st.checkbox('Show dataframe'):
    st.dataframe(pd.DataFrame(df.transpose()))



#Show resp Feature    

df = pd.read_csv("your_path_folder\%s" %filename)
df = DataFrame(df,columns=sorted(list(df.columns)))

columns = st.multiselect(
    label='Search For The Respective Feature (Can be multiple)', options=df.columns)
st.write("It Will Plot the Line Chart Of Resp Feature : ")

st.line_chart(df[columns])

st.markdown("<h4 style='text-align: center; color: grey;'>Version 1.0.1</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>@PYMETRICS</h4>", unsafe_allow_html=True)


# n=60
# if n>50 :   
#        my_slot1 = st.empty()
#        st.markdown("<h3 style='text-align: left; color: red;'>Risky</h3>", unsafe_allow_html=True)
# progress_bar = st.progress(0)
# status_text = st.empty()
# for i in range(n):

#     progress_bar.progress(i + 1)
#     status_text.text(
#         'Calculating...')

                
#     time.sleep(0.015)
# status_text.text('Risky')
# st.markdown("<h3 style='text-align: left; color: red;'>Risky</h3>", unsafe_allow_html=True)
# status_text.text('Risky')


# n=40
# if n>50 :
#        st.markdown("<h3 style='text-align: left; color: red;'>Risky</h3>", unsafe_allow_html=True)
# if n<50 :
#           st.markdown("<h3 style='text-align: left; color: Green;'>Healthy</h3>", unsafe_allow_html=True)          
# status_text = st.empty()
# progress_bar1 = st.progress(0)
# for i in range(n):

#     progress_bar1.progress(i + 1)
                
#     time.sleep(0.015)
