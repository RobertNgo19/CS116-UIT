import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set()

st.title('Classification với PCA để giảm số chiều.')

#Let's import the data from sklearn
from sklearn.datasets import load_wine
wine=load_wine()

df = pd.DataFrame(data=np.c_[wine['data'],wine['target']],columns=wine['feature_names']+['target'])
df.target=df.target.astype('int64').astype('category')

st.markdown("### Wine Datasets")
st.write(df)




from sklearn.preprocessing import StandardScaler

#Remove target columns.
x = df.loc[:,df.columns != 'target'].values
y = df.loc[:,['target']].values

#Scale the data
x= pd.DataFrame(StandardScaler().fit_transform(x))
y=pd.DataFrame(y)



from sklearn.model_selection import train_test_split,cross_val_score

option = st.selectbox(
		'Số Chiều Sau Khi Giảm:',
		('1','2','3','4','5'))
st.write('Your selected:',option)
# Create PCA object.
pca = PCA(n_components=int(option))

#Run PCA.
pComp=pca.fit_transform(x)
if option=='1':
	pca_df = pd.DataFrame(data = pComp
             , columns = ['PC 1'])
if option=='2':
	pca_df = pd.DataFrame(data = pComp
             , columns = ['PC 1','PC 2'])
if option=='3':
	pca_df = pd.DataFrame(data = pComp
             , columns = ['PC 1','PC 2', 'PC 3'],)
if option=='4':
	pca_df = pd.DataFrame(data = pComp
             , columns = ['PC 1','PC 2', 'PC 3','PC 4'],)
if option=='5':
	pca_df = pd.DataFrame(data = pComp
             , columns = ['PC 1','PC 2', 'PC 3','PC 4','PC 5'],)


st.write('Dữ liệu gốc',df.shape)
st.write('Dữ liệu sau khi PCA',pca_df.shape)


pcaclf=RandomForestClassifier(n_estimators=10,random_state=42)

pcaclf.fit(pca_df,y.values.ravel())

st.markdown('### Áp dụng xác thực chéo để đánh giá kết quả')
scores=cross_val_score(pcaclf,pca_df,y.values.ravel(),cv=5)
st.write(scores)

st.markdown('### Tính toán giá trị trung bình và độ lệch chuẩn của xác thực')
st.write('Mean',scores.mean())
st.write('Standard Dev',scores.std())






