import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

hisse_model_dict = {
    "ASELSAN": {"derin": "aselsan_derin.keras", "lstm": "aselsan_lstm.keras", "code": "ASELS.IS"},
    "AKBANK": {"derin": "akbank_derin.keras", "lstm": "akbank_lstm.keras", "code": "AKBNK.IS"},
    "EREĞLİ": {"derin": "eregli.keras", "lstm": "ergel_lstm.keras", "code": "EREGL.IS"},
    "KCHOL": {"derin": "kchol_derin.keras", "lstm": "kchol_lstm.keras", "code": "KCHOL.IS"},
    "PETKİM": {"derin": "petkim_derin.keras", "lstm": "petkim_lstm.keras", "code": "PETKM.IS"},
    "SABANCI": {"derin": "sahol_derin.keras", "lstm": "sahol_lstm.keras", "code": "SAHOL.IS"},
    "TCELL": {"derin": "turkcel_derin.keras", "lstm": "trkcel_lstm.keras", "code": "TCELL.IS"},
    "THY": {"derin": "türk_hava_yolları.keras", "lstm": "thy_lstm.keras", "code": "THYAO.IS"}
}

st.set_page_config(layout="wide")
st.title("Hisse Tahmin ve Al-Sat Sinyalleri")

st.sidebar.header("Ayarlar")
model_tipi = st.sidebar.radio("Model Tipini Seçin", ["Derin Öğrenme", "LSTM"])
selected_hisse = st.sidebar.selectbox("Lütfen bir hisse seçin:", list(hisse_model_dict.keys()))
start_date = st.sidebar.date_input("Başlangıç Tarihi", pd.to_datetime("2014-01-01"))
end_date = st.sidebar.date_input("Bitiş Tarihi", pd.to_datetime("2024-01-01"))

model_path = hisse_model_dict[selected_hisse]["derin"] if model_tipi == "Derin Öğrenme" else hisse_model_dict[selected_hisse]["lstm"]
model = load_model(model_path)
st.success(f"{model_tipi} modeli başarıyla yüklendi!")

if st.sidebar.button("Veriyi İşle"):
    if start_date < end_date:
        st.write(f"### {selected_hisse} Verileri ({model_tipi})")
        hisse_kodu = hisse_model_dict[selected_hisse]["code"]
        veri = yf.download(hisse_kodu, start=start_date, end=end_date)
        scaler = MinMaxScaler(feature_range=(0, 1))

        if model_tipi == "Derin Öğrenme":
            x = veri[["Open", "High", "Low", "Adj Close", "Volume"]].values
            x_scaled = scaler.fit_transform(x)
            tahminler = model.predict(x_scaled)
            veri["Tahmin"] = tahminler[:, 0]
        else:  # LSTM için veri hazırlığı
            scaled_data = scaler.fit_transform(veri['Close'].values.reshape(-1, 1))
            X = [scaled_data[i-100:i, 0] for i in range(100, len(scaled_data))]
            X = np.array(X).reshape(-1, 100, 1)
            tahminler = model.predict(X)
            veri = veri.iloc[100:].copy()
            veri["Tahmin"] = scaler.inverse_transform(tahminler)

        veri['Signal'] = 0
        veri.loc[veri['Tahmin'] > veri['Close'] * 1.01, 'Signal'] = 1
        veri.loc[veri['Tahmin'] < veri['Close'] * 0.99, 'Signal'] = -1

        elimizdeki_para = 100000
        hisse_adeti = 0
        toplam_deger_listesi = []
        para_listesi = [100000]
        hisse_listesi = []

        for i in range(len(veri)):
            if veri['Signal'][i] == 1:  # Alım sinyali
                adet = int(elimizdeki_para / veri['Close'][i])
                hisse_adeti += adet
                elimizdeki_para -= adet * veri['Close'][i]
            elif veri['Signal'][i] == -1:  # Satış sinyali
                elimizdeki_para += hisse_adeti * veri['Close'][i]
                hisse_adeti = 0

            toplam_deger = elimizdeki_para + hisse_adeti * veri['Close'][i]
            toplam_deger_listesi.append(toplam_deger)
            para_listesi.append(elimizdeki_para)
            hisse_listesi.append(hisse_adeti)
        st.write("### Alım-Satım Sinyalleri")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=veri.index, y=veri['Close'], name="Kapanış Fiyatı", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=veri[veri['Signal'] == 1].index, y=veri['Close'][veri['Signal'] == 1],
                                 mode="markers", name="Al Sinyali", marker=dict(color='green', symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=veri[veri['Signal'] == -1].index, y=veri['Close'][veri['Signal'] == -1],
                                 mode="markers", name="Sat Sinyali", marker=dict(color='red', symbol='triangle-down')))
        st.plotly_chart(fig, use_container_width=True)
        st.write("### Finansal Sonuçlar")
        st.write(f"**Başlangıç Sermayemiz:** 100000 ₺")
        toplam_kazanc = toplam_deger_listesi[-1] - 100000
        st.write(f"**Toplam Kazanç:** {toplam_kazanc:.2f} ₺")
        st.write(f"**Son Toplam Para:** {toplam_deger_listesi[-1]:.2f} ₺")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("### Hisse Adedi Değişimi")
            fig1 = plt.figure(figsize=(14, 6))
            plt.bar(range(len(hisse_listesi)), hisse_listesi, color='black')
            plt.xlabel("Gün")
            plt.ylabel("Hisse Adedi")
            st.pyplot(fig1)

        with col2:
            st.write("### Elimizdeki Para Değişimi")
            fig2 = plt.figure(figsize=(14, 6))
            plt.plot(para_listesi, color="orange")
            plt.axhline(100000, color="red", linestyle="--")
            plt.xlabel("Gün")
            plt.ylabel("Para (TL)")
            st.pyplot(fig2)

        with col3:
            st.write("### Toplam Değer Değişimi")
            fig3 = plt.figure(figsize=(14, 6))
            plt.plot(toplam_deger_listesi, color="blue")
            plt.axhline(100000, color="red", linestyle="--")
            plt.xlabel("Gün")
            plt.ylabel("Toplam Değer (TL)")
            st.pyplot(fig3)

        baslangic_para_altin = 100000
        baslangic_para_gumus = 100000
        usdt_tl = 35

        st.write("### Altın ve Gümüş Fiyatları")

        gold_symbol = "GC=F"
        gold_data = yf.download(gold_symbol, start="2014-01-01", end="2024-01-01")
        gold_price_try = gold_data['Close'] * usdt_tl
        ilk_gun_gold = gold_price_try.iloc[0]
        son_gun_gold = gold_price_try.iloc[-1]
        gold_miktar = baslangic_para_altin / ilk_gun_gold
        altin_son_deger = gold_miktar * son_gun_gold
        altin_kar = altin_son_deger - baslangic_para_altin
        st.write(f"Altın için toplam kâr: {altin_kar:.2f} TL")
        plt.figure(figsize=(14, 6))
        plt.plot(gold_price_try.index, gold_price_try, color="blue", label="Altın Fiyatı (TL)")
        plt.axhline(ilk_gun_gold, color="red", linestyle="--", 
                    label=f"ALIM fİYATI: {ilk_gun_gold:.2f} TL")
        plt.axhline(son_gun_gold, color="purple", linestyle="--", 
                    label=f"sATIŞ fİYATI: {son_gun_gold:.2f} TL")
        plt.title("Altın Fiyatları ve Değer Değişimi")
        plt.xlabel("Tarih")
        plt.ylabel("Fiyat (TL)")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        silver_symbol = "SI=F"
        silver_data = yf.download(silver_symbol, start="2014-01-01", end="2024-01-01")
        silver_price_try = silver_data['Close'] * usdt_tl
        ilk_gun_silver = silver_price_try.iloc[0]
        son_gun_silver = silver_price_try.iloc[-1]
        silver_miktar = baslangic_para_gumus / ilk_gun_silver
        gumus_son_deger = silver_miktar * son_gun_silver
        gumus_kar = gumus_son_deger - baslangic_para_gumus
        st.write(f"Gümüş için toplam kâr: {gumus_kar:.2f} TL")
        plt.figure(figsize=(14, 6))
        plt.plot(silver_price_try.index, silver_price_try, color="green", label="Gümüş Fiyatı (TL)")
        plt.axhline(ilk_gun_silver, color="red", linestyle="--", 
                    label=f"ALIM fİYATI: {ilk_gun_silver:.2f} TL")
        plt.axhline(son_gun_silver, color="purple", linestyle="--", 
                    label=f"sATIŞ fİYATI: {son_gun_silver:.2f} TL")
        plt.title("Gümüş Fiyatları ve Değer Değişimi")
        plt.xlabel("Tarih")
        plt.ylabel("Fiyat (TL)")
        plt.legend()
        plt.grid()
        st.pyplot(plt)
    else:
        st.error("Lütfen geçerli bir tarih aralığı seçin!")
