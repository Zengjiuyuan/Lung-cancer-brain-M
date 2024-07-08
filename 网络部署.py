import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import streamlit as st
import joblib
import os
import chardet

# 读取训练集数据
@st.cache_data
def load_data():
    try:
        # 使用相对路径
        file_path = os.path.join(os.path.dirname(__file__), '训练集.csv')
        
        # 自动检测文件编码
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        st.write(f"Detected encoding: {encoding}")  # 显示检测到的编码
        
        # 尝试使用检测到的编码读取文件
        df = pd.read_csv(file_path, encoding=encoding)
        
        # 如果成功，返回数据框
        return df
    except Exception as e:
        st.error(f"无法读取CSV文件: {e}")
        return None

train_data = load_data()

if train_data is None:
    st.error("无法加载数据，请检查CSV文件。")
    st.stop()

# 显示数据框的前几行，以确认是否正确读取
st.write("Data Preview:")
st.write(train_data.head())

# ... [其余代码保持不变] ...

# 分离输入特征和目标变量
X = train_data[['Age', 'Grade',
                   'Histology', 'T', 'N', 'Surgery', 'Radiation', 'Chemotherapy']]
y = train_data['brain metastasis']

# 创建并训练GBM模型
gbm_model = GradientBoostingClassifier()
gbm_model.fit(X, y)


# 特征映射
feature_order = [
    'Age', 'Grade', 'Histology', 'T',
    'N', 'Surgery', 'Radiation', 'Chemotherapy'
]
class_mapping = {0: "No brain metastasis", 1: "Lung cancer brain metastasis"}
Age_mapper = {"＜68 years": 0, "68-79 years": 1, "＞79 years": 2}
Grade_mapper = {"grade I": 0, "grade II": 1, "grade III": 2, "grade IV": 3}
Histology_mapper = {"adenocarcinoma": 0, "Squamous carcinoma": 1, "Small cell carcinoma": 2, "Large cell carcinoma": 3}
T_mapper = {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
N_mapper = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}
Surgery_mapper = {"NO": 0, "Yes": 1}
Radiation_mapper = {"NO": 0, "Yes": 1}
Chemotherapy_mapper = {"NO": 0, "Yes": 1}



# 预测函数
def predict_brain_metastasis(Age, Grade,
                            Histology, T, N, Surgery, Radiation, Chemotherapy):
    input_data = pd.DataFrame({
        'Age': [Age_mapper[Age]],
        'Grade': [Grade_mapper[Grade]],
        'Histology': [Histology_mapper[Histology]],
        'T': [T_mapper[T]],
        'N': [N_mapper[N]],
        'Surgery': [Surgery_mapper[Surgery]],
        'Radiation': [Radiation_mapper[Radiation]],
        'Chemotherapy': [Chemotherapy_mapper[Chemotherapy]],
    }, columns=feature_order)

    prediction = gbm_model.predict(input_data)[0]
    probability = gbm_model.predict_proba(input_data)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability

# 创建Web应用程序
st.title("GBM Model Predicting Barin Metastasis of Lung Cancer")
st.sidebar.write("Variables")

Age = st.sidebar.selectbox("Age", options=list(Age_mapper.keys()))
Grade = st.sidebar.selectbox("Grade", options=list(Grade_mapper.keys()))
Histology = st.sidebar.selectbox("Histology", options=list(Histology_mapper.keys()))
T = st.sidebar.selectbox("T", options=list(T_mapper.keys()))
N = st.sidebar.selectbox("N", options=list(N_mapper.keys()))
Surgery = st.sidebar.selectbox("Surgery", options=list(Surgery_mapper.keys()))
Radiation = st.sidebar.selectbox("Radiation", options=list(Radiation_mapper.keys()))
Chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(Chemotherapy_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_brain_metastasis(Age, Grade,
                                                     Histology, T, N, Surgery, Radiation, Chemotherapy )

    st.write("Class Label: ", prediction)  # 结果显示在右侧的列中
    st.write("Probability of developing brain metastasis: ", probability)  # 结果显示在右侧的列中


