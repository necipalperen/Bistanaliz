import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from keras.models import load_model
from joblib import load
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

hisse_model_dict = {
    "ASELSAN": {"derin": "aselsan_derin.keras", "lstm": "aselsan_lstm.keras","Arima":"aselsan_arima_model.pkl", "code": "ASELS.IS"},
    "AKBANK": {"derin": "akbank_derin.keras", "lstm": "akbank_lstm.keras","Arima":"AKBNK_arima.pkl","code": "AKBNK.IS"},
    "EREĞLİ": {"derin": "eregli.keras", "lstm": "ergel_lstm.keras","Arima":"eregli_arima_model.pkl", "code": "EREGL.IS"},
    "KCHOL": {"derin": "kchol_derin.keras", "lstm": "kchol_lstm.keras","Arima":"kchol_arima_model.pkl", "code": "KCHOL.IS"},
    "PETKİM": {"derin": "petkim_derin.keras", "lstm": "petkim_lstm.keras","Arima":"PETKİM_arima_model.pkl", "code": "PETKM.IS"},
    "SABANCI": {"derin": "sahol_derin.keras", "lstm": "sahol_lstm.keras","Arima":"SAHOL_arima_model.pkl", "code": "SAHOL.IS"},
    "TCELL": {"derin": "turkcel_derin.keras", "lstm": "trkcel_lstm.keras","Arima":"turkcel_arima_model.pkl", "code": "TCELL.IS"},
    "GARAN": {"derin": "garan_derin.keras", "lstm": "garan_lstm.keras","Arima":"garan_arima_model.pkl", "code": "GARAN.IS"},
    "KRDMD": {"derin": "kardemir_derin.keras", "lstm": "kardemir_lstm.keras","Arima":"krdmd_arima_model.pkl", "code": "KRDMD.IS"},

}

st.set_page_config(layout="wide")
st.title("Hisse Tahmin ve Al-Sat Sinyalleri")
st.subheader("Bu çalışma, Necip Alperen Tuğan'ın doktora tez çalışması kapsamında gerçekleştirilmiştir. Sunulan veriler ve modeller yalnızca akademik amaçlara hizmet etmekte olup, yatırım tavsiyesi niteliği taşımamaktadır.")
st.sidebar.header("Ayarlar")
model_tipi = st.sidebar.radio("Model Tipini Seçin", ["Derin Öğrenme", "LSTM","Arima"])
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
            veri['Signal'] = 0
            veri.loc[veri['Tahmin'] > veri['Close'] * 1.01, 'Signal'] = 1
            veri.loc[veri['Tahmin'] < veri['Close'] * 0.99, 'Signal'] = -1
        elif model_tipi == "LSTM":
            scaled_data = scaler.fit_transform(veri['Close'].values.reshape(-1, 1))
            X = [scaled_data[i-100:i, 0] for i in range(100, len(scaled_data))]
            X = np.array(X).reshape(-1, 100, 1)
            tahminler = model.predict(X)
            veri = veri.iloc[100:].copy()
            veri["Tahmin"] = scaler.inverse_transform(tahminler)
            veri['Signal'] = 0
            veri.loc[veri['Tahmin'] > veri['Close'] * 1.01, 'Signal'] = 1
            veri.loc[veri['Tahmin'] < veri['Close'] * 0.99, 'Signal'] = -1
        else:
            def modeli_yukle_ve_tahmin_yap(hisse_senedi, baslangic_tarihi, bitis_tarihi, model_dosyasi):
                veri = yf.download(hisse_senedi, start=baslangic_tarihi, end=bitis_tarihi)
                veri = veri[['Close']].dropna()  
                model_fit = load(model_dosyasi)
                tahminler = model_fit.predict(start=0, end=len(veri)-1)
                veri["Tahmin"] = tahminler
                return veri
            hisse_kodu = hisse_model_dict[selected_hisse]["code"]
            arima_model_dosyasi = hisse_model_dict[selected_hisse]["Arima"]
            try:
                veri = modeli_yukle_ve_tahmin_yap(hisse_kodu, start_date, end_date, arima_model_dosyasi)
                veri["Signal"] = 0
                veri.loc[veri["Tahmin"] > veri["Close"], "Signal"] = 1
                veri.loc[veri["Tahmin"] < veri["Close"], "Signal"] = -1
            except Exception as e:
                st.error(f"ARIMA modeli çalıştırılamadı: {e}")

        elimizdeki_para = 100000
        hisse_adeti = 0
        toplam_deger_listesi = []
        para_listesi = [100000]
        hisse_listesi = []

        for i in range(len(veri)):
            if veri['Signal'][i] == 1: 
                adet = int(elimizdeki_para / veri['Close'][i])
                hisse_adeti += adet
                elimizdeki_para -= adet * veri['Close'][i]
            elif veri['Signal'][i] == -1:  
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
        st.write("### Model Tahmini ve Gerçek Değerler")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=veri.index,
            y=veri['Close'],
            name="Kapanış Fiyatı",
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=veri.index,
            y=veri['Tahmin'],
            name="Model Tahmini",
            line=dict(color='orange', dash='dot')
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Finansal Sonuçlar")
        st.write(f"**Başlangıç Sermayemiz:** 100000 ₺")
        toplam_kazanc = toplam_deger_listesi[-1] - 100000
        st.write(f"**Toplam Kazanç:** {toplam_kazanc:.2f} ₺")
        st.write(f"**Son Toplam Para:** {toplam_deger_listesi[-1]:.2f} ₺")

        col1, col2, col3,col4= st.columns(4)

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
            plt.grid()
            st.pyplot(fig3)
        with col4:
            st.write("### Alım, Bekleme, Satış Sinyal Dağılımı")
            fig4 = plt.figure(figsize=(14, 6))
            veri['sinyal_str'] = veri['Signal'].map({1: '1 Al', 0: '0 Bekle', -1: '-1 Sat'})
            sinyal_sayilari = veri['sinyal_str'].value_counts()
            bars = plt.bar(sinyal_sayilari.index, sinyal_sayilari.values, color=['green', 'gray', 'red'])
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2, 
                    height, 
                    f"{int(height)}",  
                    ha="center", va="bottom", fontsize=10  
                )
            plt.xlabel("Sinyal")
            plt.ylabel("Sayı")
            plt.title("Alım, Bekleme ve Satış Sinyallerinin Dağılımı")
            st.pyplot(fig4)

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
        data = {
            "Yıl": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            "Yıl Başı Değeri": [100000, 111250, 119593.75, 128055.28, 138299.70, 171491.64, 192070.64, 208405.89, 243843.84, 277968.77],
            "Yıl Sonu Değeri": [111250, 119593.75, 128055.28, 138299.70, 171491.64, 192070.64, 208405.89, 243843.84, 277968.77, 303046.96]
        }
        df = pd.DataFrame(data)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Veri Tablosu:")
            st.dataframe(df)
        
        with col2:
            st.write("\n")
            st.write("""
            ### Faiz Birikimi Hesaplama Detayları:
            - **2014 Ocak – 2014 Sonu**: 100,000 TL * (1 + 0.1125) = 111,250 TL  
            - **2015 Ocak – 2015 Sonu**: 111,250 TL * (1 + 0.075) = 119,593.75 TL  
            - **2016 Ocak – 2016 Sonu**: 119,593.75 TL * (1 + 0.075) = 128,055.28 TL  
            - **2017 Ocak – 2017 Sonu**: 128,055.28 TL * (1 + 0.08) = 138,299.70 TL  
            - **2018 Ocak – 2018 Sonu**: 138,299.70 TL * (1 + 0.24) = 171,491.64 TL  
            - **2019 Ocak – 2019 Sonu**: 171,491.64 TL * (1 + 0.12) = 192,070.64 TL  
            - **2020 Ocak – 2020 Sonu**: 192,070.64 TL * (1 + 0.085) = 208,405.89 TL  
            - **2021 Ocak – 2021 Sonu**: 208,405.89 TL * (1 + 0.17) = 243,843.84 TL  
            - **2022 Ocak – 2022 Sonu**: 243,843.84 TL * (1 + 0.14) = 277,968.77 TL  
            - **2023 Ocak – 2023 Sonu**: 277,968.77 TL * (1 + 0.09) = 303,046.96 TL  
            """)
        st.write("### Faiz Birikimi Sonuçları")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df["Yıl"], df["Yıl Sonu Değeri"], marker='o', label="Yıl Sonu Değeri")
        ax.set_title("Faiz Birikimi Sonuçları Yıl Sonu Değerleri", fontsize=16)
        ax.set_xlabel("Yıl", fontsize=12)
        ax.set_ylabel("Değer (TL)", fontsize=12)
        ax.legend()
        ax.grid()
        st.pyplot(fig)
    else:
        st.error("Lütfen geçerli bir tarih aralığı seçin!")
