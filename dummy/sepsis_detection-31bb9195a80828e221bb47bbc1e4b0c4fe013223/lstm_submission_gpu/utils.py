from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

def get_data_from_file(inputFile):
	data = pd.read_csv(inputFile, delimiter='|', header=0)
	data = data.interpolate()
	data = data.fillna(method ='ffill')
	data = data.fillna(data.mean())
	data = data.fillna(0)

	return data

def d_to_features(training_data, i, mean_values=None, std_values=None, min_values=None, max_values=None):
	data = training_data.loc[i, training_data.columns!='SepsisLabel'].values
	if i > 0:
		prev_data = training_data.loc[i-1, training_data.columns!='SepsisLabel'].values
	else:
		prev_data = training_data.loc[i, training_data.columns!='SepsisLabel'].values
	if i > 1:
		prev2_data = training_data.loc[i-2, training_data.columns!='SepsisLabel'].values
	else:
		prev2_data = prev_data
	if i > 2:
		prev3_data = training_data.loc[i-3, training_data.columns!='SepsisLabel'].values
	else:
		prev3_data = prev2_data

	if i > 3:
		prev4_data = training_data.loc[i-4, training_data.columns!='SepsisLabel'].values
	else:
		prev4_data = prev3_data

	if i > 4:
		prev5_data = training_data.loc[i-5, training_data.columns!='SepsisLabel'].values
	else:
		prev5_data = prev4_data

	if i + 1 < training_data.shape[0]:
		next_data = training_data.loc[i+1, training_data.columns!='SepsisLabel'].values
	else:
		next_data = training_data.loc[i, training_data.columns!='SepsisLabel'].values
	if i + 2 < training_data.shape[0]:
		next2_data = training_data.loc[i+2, training_data.columns!='SepsisLabel'].values
	else:
		next2_data = next_data
	if i + 3 < training_data.shape[0]:
		next3_data = training_data.loc[i+3, training_data.columns!='SepsisLabel'].values
	else:
		next3_data = next2_data
	if i + 4 < training_data.shape[0]:
		next4_data = training_data.loc[i+4, training_data.columns!='SepsisLabel'].values
	else:
		next4_data = next3_data
	if i + 5 < training_data.shape[0]:
		next5_data = training_data.loc[i+5, training_data.columns!='SepsisLabel'].values
	else:
		next5_data = next4_data




	# selected_features = [0,1,2,3,4,5,6,7,8,9,10,11,12,21,34,35,36,37,38,39]
	selected_features = range(40)
	features ={}
	for j in selected_features:


		# features[str(j)] = str(int((data[j] - min_values[j])*4/(max_values[j]-min_values[j]+0.00001)))
		# features['change_'+str(j)] = data[j] - prev_data[j]
		# features['change_prev_'+str(j)] = prev_data[j] -prev2_data[j]
		features['is_increase_'+str(j)] = (data[j] - prev_data[j] > 0)
		features['prev_increase_'+str(j)] = (prev_data[j] - prev2_data[j]) > 0

		features['prev2_increase_'+str(j)] = (prev2_data[j] - prev3_data[j]) > 0
		features['prev3_increase_'+str(j)] = (prev3_data[j] - prev4_data[j]) > 0

		features['prev4_increase_'+str(j)] = (prev4_data[j] - prev5_data[j]) > 0

		# features['current_is_max_'+str(j)] = (data[j] == np.max([data[j], prev_data[j], prev2_data[j], prev3_data[j], prev4_data[j], prev5_data[j]]))

		features['next_increase_'+str(j)]= (next_data[j] - data[j] > 0)
		features['next2_increase_'+str(j)]= (next2_data[j] - next_data[j] > 0)
		features['next3_increase_'+str(j)]= (next3_data[j] - next2_data[j] > 0)
		features['next4_increase_'+str(j)]= (next4_data[j] - next3_data[j] > 0)
		features['next5_increase_'+str(j)]= (next5_data[j] - next4_data[j] > 0)

		# features['current_is_abnormal_'+str(j)] = (abs(data[j] - mean_values[j]) > 3* std_values[j])

		if j == 0:
			features['HR_high'] = (data[j] >= 100)
			features['HR_low'] = (data[j] < 60)
			# features['HR_high_prev'] = (prev_data[j] > 90)
			# features['HR_high_prev2'] = (prev2_data[j] > 90)
			# features['HR_high_prev3'] = (prev3_data[j] > 90)
			# features['HR_high_prev4'] = (prev4_data[j] > 90)
			# features['HR_high_prev5'] = (prev5_data[j] > 90)
		elif j == 1:
			features['Low_O2_Sat'] = (data[j] < 95)
			features['High_O2_Sat'] = (data[j] > 100)

		elif j == 2:
			features['Temp_high']=(data[j] > 37.2)
			features['Temp_low'] = (data[j] < 36.1)
			# features['Temp_high_prev']=(prev_data[j] > 38)
			# features['Temp_low_prev'] = (prev_data[j] < 36)
			# features['Temp_high_prev2']=(prev2_data[j] > 38)
			# features['Temp_low_prev2'] = (prev2_data[j] < 36)
			# features['Temp_high_prev3']=(prev3_data[j] > 38)
			# features['Temp_low_prev3'] = (prev3_data[j] < 36)
			# features['Temp_high_prev4']=(prev4_data[j] > 38)
			# features['Temp_low_prev4'] = (prev4_data[j] < 36)
			# features['Temp_high_prev5']=(prev5_data[j] > 38)
			# features['Temp_low_prev5'] = (prev5_data[j] < 36)
		

		elif j ==3:
			# sbp_data = training_data.loc[:i, training_data.columns=='SBP'].values
			# max_sbp = np.max(sbp_data)
			# min_sbp = np.min(sbp_data)

			# features['Very_High_sbp_in_past'] = (max_sbp >= 150)
			# features['Very_Low_sbp_in_past'] = (min_sbp <= 90)

			features['Curent_very_low_normal_sbp'] = (data[j] < 120)
			features['Curent_low_normal_sbp'] = (data[j] >= 120 and data[j] < 130)
			features['Curent_high_normal_sbp'] = (data[j] >= 130 and data[j] < 140)
			features['Curent_high_sbp'] = (data[j] >= 140 and data[j] < 150)
			features['Curent_very_high_sbp'] = (data[j] >= 150)



		elif j == 4:
			# map_data = training_data.loc[:i, training_data.columns=='MAP'].values
			# max_map = np.max(map_data)
			# min_map = np.min(map_data)
			features['High_MAP'] = (data[j] > 100)
			features['Low_MAP'] = (data[j] < 70)
		elif j == 5:
			# dbp_data = training_data.loc[:i, training_data.columns=='DBP'].values
			# max_dbp = np.max(dbp_data)
			# min_dbp = np.min(dbp_data)
			features['High_DBP'] = (data[j] >= 80)
			features['Low_DBP'] = (data[j] < 55)

		elif j ==6:
			features['High_Resp'] = (data[j] > 16)
			features['Low_Resp'] = (data[j] < 12)
		elif j ==7:
			features['High_EtCo2'] = (data[j] > 45)
			features['Low_EtCo2'] = (data[j] < 35)
		elif j ==8:
			features['High_Base_Excess'] = (data[j] > 3)
			features['Low_Base_Excess'] = (data[j] <-3)
		elif j ==9:
			features['High_HCO3'] = (data[j] > 30)
			features['Low_HCO3'] = (data[j] < 23)
		elif j ==10:#Question
			features['Low_FiO2'] = (data[j] < 0.21)
			features['High_Fi02'] = (data[j] >= 0.21)
		elif j == 11:
			features['High_PH'] = (data[j] > 7.45)
			features['Low_PH'] = (data[j] < 7.35)

		elif j == 12:
			features['High_PaCO2'] = (data[j] > 42)
			features['Low_PaCo2'] = (data[j] < 38)
			# features['PaCO2_low_prev'] = (prev_data[j] < 32)
			# features['PaCO2_low_prev2'] = (prev2_data[j] < 32)
			# features['PaCO2_low_prev3'] = (prev3_data[j] < 32)
			# features['PaCO2_low_prev4'] = (prev4_data[j] < 32)
			# features['PaCO2_low_prev5'] = (prev5_data[j] < 32)

		elif j == 13:
			features['High_SaO2'] = (data[j] > 97)
			features['Low_SaO2'] = (data[j] < 93)
		elif j == 14:
			features['High_AST'] = (data[j] > 200)
			features['Low_AST'] = (data[j] < 100)

		elif j==15:
			features['High_BUN'] = (data[j] > 20)
			features['Low_BUN'] = (data[j] < 7)
		elif j ==16:
			features['High_ALP'] = (data[j] > 147)
			features['Low_ALP'] = (data[j] < 44)
		elif j == 17:
			features['High_Calxi'] = (data[j] > 10.2)
			features['Low_Calxi'] = (data[j] < 8.5)
		elif j == 18:
			features['High_Chloride'] = (data[j] > 106)
			features['Low_Chloride'] = (data[j] < 98)
		elif j == 19:
			features['High_Creatinine'] = (data[j] > 1.2)
			features['Low_Creatinine'] = (data[j] < 0.6)
		elif j == 20:
			features['High_Bilirubin_direct'] = (data[j] > 0.3)
			# features['Low_Bilirubin_direct'] = (data[j] < 0.1)
		elif j == 21:
			features['High_Glucose'] = (data[j] > 140)
			features['Low_Glucose'] = (data[j] < 70)
		elif j == 22:
			features['High_Lactate'] = (data[j] > 18.2)
			features['Low_lactate'] = (data[j] < 9.1)
		elif j==23:
			features['High_Mage'] = (data[j] > 10) #Chua chac
			features['Low_Mage'] = (data[j] < 7)

		elif j == 24:
			features['High_Phosphate'] = (data[j] > 4.5)
			features['Low_Phosphate'] = (data[j] < 2.5)
		elif j == 25:
			features['High_Potasisium'] = (data[j] > 5)
			features['Low_Potasisium'] = (data[j] < 3.5)
		elif j == 26:
			features['High_total_bilirubin'] = (data[j] > 1.2)
			features['Low_total_bilirubin'] = (data[j] < 0.1)
		elif j== 27:
			features['High_troponin'] = (data[j] > 0.4)
		elif j==28:
			if data[35] == 1:
				features['High_Hct'] = (data[j] > 52)
				features['Low_Hct'] = (data[j] < 45)
			else:
				features['High_Hct'] = (data[j] > 48)
				features['Low_Hct'] = (data[j] < 37)
		elif j==29:
			if data[35] == 1:
				features['High_Hgb'] = (data[j] > 17.1)
				features['Low_Hgb'] = (data[j] < 13.8)
			else:
				features['High_Hgb'] = (data[j] > 15.1)
				features['Low_Hgb'] = (data[j] < 12.1)
		elif j ==30:
			features['High_PTT'] = (data[j] > 35)
			features['Low_PTT'] = (data[j] < 25)
		elif j ==31:
			features['High_WBC'] = (data[j] > 110) #check lai
			features['Low_WBC'] = (data[j] < 45)
		elif j == 32:
			features['High_Fibrinogen'] = (data[j] > 400)
			features['Low_Fibrinogen'] = (data[j] < 150)
		elif j ==33:
			features['High_Platelets'] = (data[j] > 4500) #check lai
			features['Low_Platelets'] = (data[j] < 1500)


		elif j==34:
			features['Age_high'] = (data[j] > 18)
			features['Age_very_high'] = (data[j] > 50)
			features['Age_too_high'] = (data[j] > 70)
		elif j==35:
			features['Is_man'] = (data[j] ==1)

			# features['Age_high_prev'] = (prev_data[j] > 18)
			# features['Age_high_prev2'] = (prev2_data[j] > 18)
			# features['Age_high_prev3'] = (prev3_data[j] > 18)
			# features['Age_high_prev4'] = (prev4_data[j] > 18)
			# features['Age_high_prev5'] = (prev5_data[j] > 18)
		elif j == 39:
			features['ICULOS'] = (data[j] <= 1)
			# features['ICULOS_prev'] == (prev_data[j] <= 1)
			# features['ICULOS_prev2'] == (prev2_data[j] <= 1)
			# features['ICULOS_prev3'] == (prev3_data[j] <= 1)
			# features['ICULOS_prev4'] == (prev4_data[j] <= 1)
			# features['ICULOS_prev5'] == (prev5_data[j] <= 1)



		# features[str(i) +'_prev'] = prev_data[i]

	return features
def d_to_label(training_data, i):
	label = training_data.loc[i, 'SepsisLabel']
	return str(label)
def data_to_features(training_data, mean_values=None, std_values=None, min_values=None, max_values=None):

	return [d_to_features(training_data, i,mean_values, std_values, min_values, max_values) for i in range(training_data.shape[0])]

def data_to_labels(training_data):
	return [d_to_label(training_data, i) for i in range(training_data.shape[0])]


def prepare_features_crf(all_training_data, is_training=True):

	# mean_values, std_values = compute_mean_std_normal_values(all_training_data)
	# min_values, max_values = compute_min_max_normal_values(all_training_data)


	mean_values, std_values = None, None
	min_values, max_values = None, None
	X_train = [data_to_features(training_data, mean_values, std_values, min_values, max_values) for training_data in all_training_data]
	if is_training:
		y_train = [data_to_labels(training_data) for training_data in all_training_data]
	else:
		y_train = None
	return X_train, y_train

def prepare_input_for_lstm_crf(all_training_data, is_training=True):
	all_sequences = []
	all_labels = []
	for i, training_data in enumerate(all_training_data):
		s = training_data.loc[:,training_data.columns!='SepsisLabel'].values


		if is_training:
			l = training_data['SepsisLabel']
		else:
			l = 0
		all_sequences.append(s)
		all_labels.append(l)
	return all_sequences, all_labels