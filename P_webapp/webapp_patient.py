import streamlit as st
import pandas as pd
import numpy as np
import time
from pandas import DataFrame
import os
st.write("""
# PATIENT'S DASHBOARD

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
image = Image.open('src\P_webapp\src\db_p1.jpg')
st.image(image,use_column_width=True)

#Image db discription
st.markdown("<h3 style='text-align: left; color: teal;'>Discription About Features</h3>", unsafe_allow_html=True)
from PIL import Image
image = Image.open('src\P_webapp\src\ds_p.png')
st.image(image,use_column_width=True)

#Folder Path Selector

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path+"/data")
    selected_filename = st.selectbox('Select a respective patient file :', filenames)
    return os.path.join(folder_path+"/data", selected_filename)

filename = file_selector()
st.write(""" ## Patient selected : `%s`""" % filename[len(filename)-11:len(filename)-4])

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






df = pd.read_csv("your_path_folder\%s" %filename)

n=int(df.iloc[-1]['HR'])
n1=int(df.iloc[-1]['Resp'])
n2=int(df.iloc[-1]['MAP'])
n3=int(df.iloc[-1]['WBC'])
n4=int(df.iloc[-1]['Platelets'])
n5=int(df.iloc[-1]['Hgb'])
#st.write(n)

#Scaling Function

def scaling(maxm , minm , n):
 return( (n - minm) / (maxm - minm) ) * (100)

# h_rate=scaling(100,60,n)
# print(h_rate)

# resp=scaling(40,10,20)
# print(resp)

# mapp=scaling(110,70,80)
# print(mapp)

# wbc=scaling(15,5,10)
# print(wbc)

# plat=scaling()
# print(plat)

# hgb=scaling()
# print(hgb)
x=0
#n=90
#n=60

m=int(scaling(150,60,n))
st.markdown("<h3 style='text-align: left; color: teal;'>Heart Rate</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: grey;'>Your heart rate, or pulse, is the number of times your heart beats per minute </h4>", unsafe_allow_html=True)
progress_bar = st.progress(0)
st.markdown("<h4 style='text-align: left; color: teal;'>Clinical analysis :</h4>", unsafe_allow_html=True)
status_text = st.empty()



for i in range(m):

    progress_bar.progress(i + 1)
    status_text.text(
        'Fetching Respective Values...')
    time.sleep(0.015)
if n>106 or n<71 :   
       my_slot1 = st.empty()
       status_text.text('At Risk')
       st.write("Normal Range 71 to 106 Your's is :")
       st.write(n)
       x=x+1
else :   
       my_slot1 = st.empty()
       status_text.text('Healthy')
       x=x+1       


#n=30
m1=int(scaling(40,10,n1))
st.markdown("<h3 style='text-align: left; color: teal;'>Respiration</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: grey;'>A person's respiratory rate is the number of breaths you take per minute.</h4>", unsafe_allow_html=True)
progress_bar = st.progress(0)
st.markdown("<h4 style='text-align: left; color: teal;'>Clinical analysis :</h4>", unsafe_allow_html=True)
status_text = st.empty()


for i in range(m1):

    progress_bar.progress(i + 1)
    status_text.text(
        'Fetching Respective Values...')
    time.sleep(0.015)
if n1<30 and n1>20 :   
       my_slot1 = st.empty()
       status_text.text('Healthy')
else :   
       my_slot1 = st.empty()
       status_text.text('At Risk')  
       st.write("Normal Range 20 to 30 Your's is :")
       st.write(n1) 
       x=x+1

#n=30
m2=int(scaling(110,0,n2))
st.markdown("<h3 style='text-align: left; color: teal;'>MAP</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: grey;'>MAP, or mean arterial pressure, is defined as the average pressure in a patient's arteries during one cardiac cycle</h4>", unsafe_allow_html=True)
progress_bar = st.progress(0)
st.markdown("<h4 style='text-align: left; color: teal;'>Clinical analysis :</h4>", unsafe_allow_html=True)
status_text = st.empty()


for i in range(m2):

    progress_bar.progress(i + 1)
    status_text.text(
        'Fetching Respective Values...')
    time.sleep(0.015)
if n2>9 or n2<16 :   
       my_slot1 = st.empty()
       status_text.text('Healthy')
else :   
       my_slot1 = st.empty()
       status_text.text('At Risk') 
       st.write("Normal Range 9 to 16 Your's is :")
       st.write(n2)  
       x=x+1

#n=30
m3=int(scaling(15,0,n3))
st.markdown("<h3 style='text-align: left; color: teal;'>WBC</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: grey;'>A WBC count is a blood test to measure the number of white blood cells (WBCs) in the blood</h4>", unsafe_allow_html=True)
progress_bar = st.progress(0)
st.markdown("<h4 style='text-align: left; color: teal;'>Clinical analysis :</h4>", unsafe_allow_html=True)
status_text = st.empty()


for i in range(m3):

    progress_bar.progress(i + 1)
    status_text.text(
        'Fetching Respective Values...')
    time.sleep(0.015)
if n3>4 or n<10 :   
       my_slot1 = st.empty()
       status_text.text('Healthy')
else :   
       my_slot1 = st.empty()
       status_text.text('At Risk')
       st.write("Normal Range 4 to 10 Your's is :")
       st.write(n3)   
       x=x+1

#n=30
m4=int(scaling(450,80,n4))
st.markdown("<h3 style='text-align: left; color: teal;'>Platelets</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: grey;'>A platelet count is a lab test to measure how many platelets you have in your blood</h4>", unsafe_allow_html=True)
progress_bar = st.progress(0)
st.markdown("<h4 style='text-align: left; color: teal;'>Clinical analysis :</h4>", unsafe_allow_html=True)
status_text = st.empty()


for i in range(m4):

    progress_bar.progress(i + 1)
    status_text.text(
        'Fetching Respective Values...')
    time.sleep(0.015)
if n4>150 or n4<350 :   
       my_slot1 = st.empty()
       status_text.text('Healthy')
else :   
       my_slot1 = st.empty()
       status_text.text('At Risk') 
       st.write("Normal Range 150 to 350 Your's is :")
       st.write(n4)  
       x=x+1       

#n=30
m5=int(scaling(30,0,n5))
st.markdown("<h3 style='text-align: left; color: teal;'>Hgb</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: left; color: grey;'>Hemoglobin (Hb or Hgb) is a protein in red blood cells that carries oxygen throughout the body</h4>", unsafe_allow_html=True)
progress_bar = st.progress(0)
st.markdown("<h4 style='text-align: left; color: teal;'>Clinical analysis :</h4>", unsafe_allow_html=True)
status_text = st.empty()


for i in range(m5):

    progress_bar.progress(i + 1)
    status_text.text(
        'Fetching Respective Values...')
    time.sleep(0.015)
if n5>14 or n5<17 :   
       my_slot1 = st.empty()
       status_text.text('Healthy')
else :   
       my_slot1 = st.empty()
       status_text.text('At Risk') 
       st.write("Normal Range 14 to 17 Your's is :")
       st.write(n5)  
       x=x+1                          

st.markdown("<h3 style='text-align: left; color: teal;'>Based On The Above Score Your Remarks Are :</h3>", unsafe_allow_html=True)            
status_text = st.empty()
if x>3 :
          status_text.text('Calculating...')
          status_text.text('Consult To The Docter')
else:
          status_text.text('Score Is Safe , Stay Healthy , Stay Hydrated :)')          


st.markdown("<h4 style='text-align: center; color: grey;'>Version 1.0.1</h4>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: grey;'>@PYMETRICS</h4>", unsafe_allow_html=True)


