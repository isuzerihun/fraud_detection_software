import pickle, sklearn
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Load the model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)


st.header('🚨 Fraud Detection Software')

with st.form("Fraud Detrection Form"):
    st.write('Enter details')
    
    
    time = st.text_input(label='Transaction Time', value='0')
    amount = st.text_input(label='Transaction Amount', value='0')
    
    feature_inputs = []
    columns = st.columns(4)
    
    for i in range(28):
        with columns[i % 4]:
            value = st.text_input(label=f'V{i+1}', value='0', key=f'Vector {i+1}')
            feature_inputs.append(value)

    
    if st.form_submit_button("Detect"):
        values = np.array(feature_inputs, dtype=float).reshape(1, -1)
        time = np.array(float(time)).reshape(-1, 1)
        amount = np.array(float(amount)).reshape(-1, 1)
        
        std = StandardScaler()
        min_max = MinMaxScaler()

        time_scaled = min_max.fit_transform(time)
        amount_scaled = std.fit_transform(amount)
        
  
        combined_input = np.concatenate([time_scaled[0], values[0], amount_scaled[0]]).reshape(1, -1)
        prediction = model.predict(combined_input)[0]
        
        if prediction == 0:
            st.success('✅ Transaction is unlikely to be fraudulent.')
            st.info('Recommendation: Standard transaction processing.')
        else:
            st.error('🚨 Transaction is likely fraudulent!')
            st.warning('Recommendation: Immediate investigation required.')
        
st.sidebar.info("""
    ### About This Tool
    - This is a machine learning-based fraud detection system
    - Input features are preprocessed before prediction
    - Always verify results with additional checks
    """)

    