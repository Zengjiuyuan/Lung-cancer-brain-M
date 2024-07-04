import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st
import joblib

# 读取训练集数据
train_data = pd.read_csv('D:/数据/文章/1、SEER+机器学习+列线图/3、数据分析/1、肺癌-8改9-脑转移-OS/11、网络部署/训练集.csv')

# 分离输入特征和目标变量
X = train_data[['Age', 'Race', 'Grade',
                   'Histology', 'T', 'N', 'Surgery', 'Radiation', 'Chemotherapy', 'Marriage']]
y = train_data['brain metastasis']

# 创建并训练GBM模型
gbm_model = GradientBoostingClassifier()
gbm_model.fit(X, y)


# 特征映射
feature_order = [
    'Age', 'Race', 'Grade', 'Histology', 'T',
    'N', 'Surgery', 'Radiation', 'Chemotherapy', 'Marriage'
]
class_mapping = {0: "No brain metastasis", 1: "Lung cancer brain metastasis"}
Age_mapper = {"＜70 years": 0, "70-79 years": 1, "＞79": 2}
Race_mapper = {"White": 0, "Black": 1, "Other": 2}
Grade_mapper = {"grade I": 0, "grade II": 1, "grade III": 2, "grade IV": 3}
Histology_mapper = {"adenocarcinoma": 0, "Squamous carcinoma": 1, "Small cell carcinoma": 2, "Large cell carcinoma": 3}
T_mapper = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
N_mapper = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}
Surgery_mapper = {"NO": 0, "Yes": 1}
Radiation_mapper = {"NO": 0, "Yes": 1}
Chemotherapy_mapper = {"NO": 0, "Yes": 1}
Marriage_mapper = {"NO": 0, "Yes": 1}


# 预测函数
def predict_brain_metastasis(Age, Race, Grade,
                            Histology, T, N, Surgery, Radiation, Chemotherapy, Marriage):
    input_data = pd.DataFrame({
        'Age': [Age_mapper[Age]],
        'Race': [Race_mapper[Race]],
        'Grade': [Grade_mapper[Grade]],
        'Histology': [Histology_mapper[Histology]],
        'T': [T_mapper[T]],
        'N': [N_mapper[N]],
        'Surgery': [Surgery_mapper[Surgery]],
        'Radiation': [Radiation_mapper[Radiation]],
        'Chemotherapy': [Chemotherapy_mapper[Chemotherapy]],
        'Marriage': [Marriage_mapper[Marriage]],
    }, columns=feature_order)

    prediction = gbm_model.predict(input_data)[0]
    probability = gbm_model.predict_proba(input_data)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability

# 创建Web应用程序
st.title("GBM Model Predicting Barin Metastasis of Lung Cancer")
st.sidebar.write("Variables")

Age = st.sidebar.selectbox("Age", options=list(Age_mapper.keys()))
Race = st.sidebar.selectbox("Race", options=list(Race_mapper.keys()))
Grade = st.sidebar.selectbox("Grade", options=list(Grade_mapper.keys()))
Histology = st.sidebar.selectbox("Histology", options=list(Histology_mapper.keys()))
T = st.sidebar.selectbox("T", options=list(T_mapper.keys()))
N = st.sidebar.selectbox("N", options=list(N_mapper.keys()))
Surgery = st.sidebar.selectbox("Surgery", options=list(Surgery_mapper.keys()))
Radiation = st.sidebar.selectbox("Radiation", options=list(Radiation_mapper.keys()))
Chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(Chemotherapy_mapper.keys()))
Marriage = st.sidebar.selectbox("Marriage", options=list(Marriage_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_brain_metastasis(Age, Race, Grade,
                                                     Histology, T, N, Surgery, Radiation, Chemotherapy, Marriage )

    st.write("Class Label: ", prediction)  # 结果显示在右侧的列中
    st.write("Probability of developing brain metastasis: ", probability)  # 结果显示在右侧的列中


