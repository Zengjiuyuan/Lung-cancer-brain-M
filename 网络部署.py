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


