import streamlit as st
import pandas as pd
import numpy as np
import joblib, json

with open("jupyter/models/meta.json") as f:
    meta = json.load(f)


crop_list = sorted(meta.keys())

st.title("Crop Yield Predictor")
st.markdown("Predict how much a crop will yield based on country, climate, and farming inputs.")

crop    = st.selectbox("Crop Type", crop_list)

if crop: 
    country = st.selectbox("Country", meta[crop]['countries'])
    r = meta[crop]['ranges']
    rainfall   = st.slider("Average Rainfall (mm/yr)", float(r['average_rain_fall_mm_per_year']['min']), float(r['average_rain_fall_mm_per_year']['max']), float(r['average_rain_fall_mm_per_year']['min']))
    pesticides = meta[crop]['country_pesticides'].get(country, 0.0)
    avg_temp   = st.slider("Average Temperature (°C)",float(r['avg_temp']['min']),float(r['avg_temp']['max']), float(r['avg_temp']['min']))
    field_size = st.number_input("Field Size (hectares)", min_value=0.1, value=1.0, step=0.5)

    if st.button("Predict Yield"):
        filename = crop.replace(" ", "_").replace(",", "").replace("/", "")
        model  = joblib.load(f"jupyter/models/{filename}_model.pkl")
        scaler = joblib.load(f"jupyter/models/{filename}_scaler.pkl")

        columns  = meta[crop]['columns']
        input_df = pd.DataFrame([{col: 0 for col in columns}])

        input_df['average_rain_fall_mm_per_year'] = rainfall
        input_df['pesticides_tonnes'] = np.log1p(pesticides)
        input_df['avg_temp'] = avg_temp

        country_col = f"Area_{country}"
        if country_col in input_df.columns:
            input_df[country_col] = 1

        prediction = model.predict(scaler.transform(input_df))[0]

        st.success(f"### Predicted Yield for {crop} in {country}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Per hectare",    f"{prediction/10:,.0f} kg/ha")
        col2.metric("Your total field", f"{(prediction/10) * field_size:,.0f} kg")
        col3.metric("In tonnes",      f"{(prediction/10000) * field_size:.2f} t")