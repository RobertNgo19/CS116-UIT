import streamlit as st
import pandas as pd
import numpy as np
from numpy import absolute
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from numpy import mean
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
st.title('Phân lớp với Logistic Regression .')

#Upload file csv (sau này có thể thay bằng tập dữ liệu khác dễ dàng).
uploaded_file = st.file_uploader("Upload CSV",type=['csv'])


if uploaded_file:
	df = pd.read_csv(uploaded_file)

	#Hiển thị bảng dữ liệu với file đã upload
	st.markdown("### Data preview")
	st.write(df)
	Choose_method = st.radio(	
    "Chọn phương pháp:",
   (' XGBoost Classifier', 'Logistic Regression',' Random Forest'),)
	st.markdown('### Chọn lựa input')
	with st.form(key = "my_form"):
		feature = st.multiselect(
			"Input Feature",options = df.columns,
			help = "Select which column refers to your A/B testing labels.",)
		
		result = st.multiselect("Output Predict",options = df.columns,)
		

		submit = st.form_submit_button(label='Submit')
		for i in (feature):
			x = df[i].values
			
		for j in (result):
			y = df[j].values

		x = x.reshape(-1,1)
		y = y.reshape(-1,1)


	parameters = st.selectbox("Chon he so train split:",('0.8','0.7','0.6'),)
	X_train,X_test,y_train,y_test = train_test_split(x,y,train_size = float(parameters),test_size = 1 - float(parameters),random_state=13)
	if st.button("Run"):
		if Choose_method == 'Logistic Regression':
			from sklearn.linear_model import LogisticRegression
			logmodel = LogisticRegression()
			logmodel.fit(X_train,y_train)

			predictions = logmodel.predict(X_test)
			from sklearn.metrics import accuracy_score
			result = accuracy_score(y_test,predictions)
			st.write("accuracy_score: ",result)

		elif Choose_method == 'XGBoost Classifier':
			from xgboots import XGBClassifier
			classifier = XGBClassifier() 
			classifier.fit(X_train,y_train)

			predictions = classifier.predict(X_test)

			from sklearn.mectrics import accuracy_score
			result = accuracy_score(y_test,predictions)
			st.write('accuracy_score:', result)

		elif Choose_method == 'Random Forest':
			from sklearn.ensemble import RandomForestClassifier
			clf = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
			clf.fit(X_train,y_train)

			predictions = clf.predict(X_test)

			from sklearn.mectrics import accuracy_score
			result = accuracy_score(y_test,predictions)
			st.write('accuracy_score:',result)
