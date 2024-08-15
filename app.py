import streamlit as st
import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostRegressor

df = pd.read_csv('Pricing.csv')
address = df['Address'].unique().tolist()
df = pd.get_dummies(df, columns=['Address'])
columns = df.columns

df = pd.read_csv('Pricing.csv')
df.drop('Price(USD)', axis=1, inplace=True)
df = df.drop_duplicates()
df['Area'] = df['Area'].str.replace(',','')
df['Area'] = df['Area'].astype(float)
df.dropna(inplace=True)
def batasAtas(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR

    return upper

df = df[df['Area'] < batasAtas(df['Area'])]
df = df[df['Price'] < batasAtas(df['Price'])]
df = pd.get_dummies(df, columns=['Address'])
df.replace(True, 1, inplace=True)
df.replace(False, 0, inplace=True)

with open('catboostModel.pkl', 'rb') as file:
    model = pickle.load(file)

def preprocessing(data):
    data = pd.get_dummies(data, columns=['Address'])
    data = data.replace({'True': 1, 'False': 0})
    data = data.reindex(columns=columns, fill_value=0)
    return data

# streamlit app

st.set_page_config(page_title="Project Akhir Study Group AI Lab", page_icon="🛒")

st.title(':blue[Prediction of House Price in Iran 💰] ')

st.markdown("""
### 📊 Project Akhir Study Group AI Lab Kelompok 4:
- Valentino Hartanto
- Tisee
- Tiara Sabrina

### 📈 Deskripsi: 
\nModel ini dibangun untuk melakukan regresi guna memprediksi harga rumah di Iran berdasarkan 6 parameter yang dapat diinputkan oleh pengguna. Dengan demikian, model ini akan membantu memberikan perkiraan kasar mengenai harga suatu rumah di Iran, sehingga pengguna dapat memperoleh gambaran umum tentang nilai properti di negara tersebut.
""")
st.write('')

Area = st.number_input("🔢 Enter Area", 10)
Room = st.selectbox('🔢 How Many Rooms Do You Need It?', [0,1,2,3,4,5])
Parking = st.selectbox('🔢 Do You Need a Parking Area?', ['True','False'])
Warehouse = st.selectbox('🔢 Do You Need a WareHouse?', ['True', 'False'])
Elevator = st.selectbox('🔢 Do You Need a Elevator?', ['True', 'False'])
Address = st.selectbox('🔢 Give me a specific location', address)

def predictPrice(Area, Room, Parking, Warehouse, Elevator, Address):
    data = np.array([Area, Room, Parking, Warehouse, Elevator, Address])
    dfData = pd.DataFrame([data], columns = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address'])
    predData = preprocessing(dfData)
    
    result = model.predict(predData)[0]
    rounded_result = round(result, 2)

    st.write('Perkiraan Harga Rumah: ')
    st.write(f'📍 Harga Dalam Riel: {rounded_result:,.2f}')
    st.write(f'📍 Harga Dalam USD: {(rounded_result/42000):,.2f}')
   
if st.button("🔎 Predict"):
    predictPrice(Area, Room, Parking, Warehouse, Elevator, Address)