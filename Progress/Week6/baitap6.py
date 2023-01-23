import streamlit as st
import pandas as pd
import numpy as np
from numpy import absolute
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from numpy import mean
from sklearn.linear_model import LogisticRegression
st.title('Phân lớp với Logistic Regression và đánh giá mô hình.')

#Upload file csv (sau này có thể thay bằng tập dữ liệu khác dễ dàng).
uploaded_file = st.file_uploader("Upload CSV",type=['csv'])


if uploaded_file:
	df = pd.read_csv(uploaded_file)

	#Hiển thị bảng dữ liệu với file đã upload
	st.markdown("### Data preview")
	st.write(df)
	Choose_method = st.radio(	
    "Chọn phương pháp:",
   ('Train/Test split', 'K-Fold cross validation'),)
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

	if Choose_method =='Train/Test split':
		Parameter = st.selectbox("Chọn hệ số train/test split:",
			('0.8','0.7','0.6'),)
		if st.button("Run"):
			#https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
			from sklearn.model_selection import train_test_split
			x_train,x_test,y_train,y_test= train_test_split(x,y,train_size = float(Parameter),random_state=0)

			lr = LogisticRegression()
			lr.fit(x_train,y_train)
			y_pred = lr.predict(x_test)

			from sklearn import metrics
			MSE = metrics.mean_squared_error(y_test, y_pred)
			MAE = metrics.mean_absolute_error(y_test, y_pred)
			st.write('MSE: ',MSE)
			st.write('MAE: ',MAE)
			from sklearn.metrics import f1_score
			F1_Score = f1_score(y_test, y_pred, average=None)
			st.write('F1-Score: ',F1_Score)

	elif Choose_method == 'K-Fold cross validation':
		with st.form(key = "your_form"):
			K_Fold_option = st.selectbox(
				"What number K do u wanna chose ?",
				('2','3','4','5','6','7','8','9','10'),)

			submit = st.form_submit_button(label='Run')

			from sklearn.model_selection import KFold, cross_val_score
			lr = LogisticRegression()
			cv = KFold(n_splits = int(K_Fold_option),shuffle = True,random_state=None)
			scores = cross_val_score(lr,x,y,cv = cv)
			st.write("Cross Validation Scrore:",scores)
			st.write("Average CV Score: ", scores.mean())


