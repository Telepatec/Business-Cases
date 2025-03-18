import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pickle
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import datetime
import preproc as p
import mappings as mapp

def show_predict():
    # CLAIM DETAILS
    st.header("Claim Details")
    with st.expander("Personal Information"):

        col1, col2 = st.columns(2)

        with col1:
            birth_year = st.slider(
                "Worker Birth Year",
                1900,
                2018,
                1980,
                help="Year of the worker's birth.",
            )

            zip_code = st.slider(
                "Zip Code",
                min_value = 0,
                value = 99999,
                help="Zip code of the worker.",
            )

            n_dependents = st.slider(
                "Number of Dependents",
                0,
                6,
                2,
                help="Number of dependents of the worker.",
            )


        with col2:
            avg_weekly_wage = st.number_input(
                "Average Weekly Wage (USD)",
                min_value = 0,
                value = 2828079,
                help = "Enter the average weekly wage of the worker.",
            )

            gender = st.radio(
                "Gender",
                ["M", "F", 'U/X'],
                horizontal=True,
                help="Indicate the gender of the worker.",
            )

            claim_iden = st.number_input(
                "Claim Identifier",
                min_value = 0,
                value = 2828079,
                help = "Enter the average weekly wage of the worker.",
            )


    # INCIDENT DETAILS        
    with st.expander("Incident Details"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            accident_year = st.slider(
                "Accident Year", 1966, 2021, 2015, help="Year of the accident."
            )
            accident_month = st.slider(
                "Accident Month", 1, 12, 6, help="Month of the accident."
            )

            accident_day = st.slider(
                "Accident Day", 1, 31, 6, help="Day of the accident."
            )

            part_of_body = st.selectbox(
                "Part of Body Injured",
                list(mapp.part_of_body_mapping.keys()),
                help="Select the injured body part.",
            )

            covid = st.radio(
                "Covid-19?",
                ["Yes", "No"],
                horizontal=True,
                help="Indicate if the claim is related to Covid-19.",
            )

        with col2:
            assembly_year = st.slider(
                "Assembly Year", 1966, 2021, 2015, help="Year of the assembly."
            )
            assembly_month = st.slider(
                "Assembly Month", 1, 12, 6, help="Month of the assembly."
            )

            assembly_day = st.slider(
                "Assembly Day", 1, 31, 6, help="Day of the assembly."
            )

            cause_injury = st.selectbox(
                "Cause of Injury",
                list(mapp.cause_of_injury_mapping.keys()),
                help="Select the cause of injury.",
            )


        with col3:
            c2_year = st.slider(
                "C-2 Year", 1966, 2021, 2015, help="Year of the C-2."
            )
            c2_month = st.slider(
                "C-2 Month", 1, 12, 6, help="Month of the C-2."
            )

            c2_day = st.slider(
                "C-2 Day", 1, 31, 6, help="Day of the C-2."
            )

            nature_injury = st.selectbox(
                "Nature of Injury",
                list(mapp.nature_of_injury_mapping.keys()),
                help="Select the cause of injury.",
            )

        with col4:
            c3_year = st.slider(
                "C-3 Year", 1966, 2021, 2015, help="Year of the C-3."
            )
            c3_month = st.slider(
                "C-3 Month", 1, 12, 6, help="Month of the C-3."
            )

            c3_day = st.slider(
                "C-3 Day", 1, 31, 6, help="Day of the C-3."
            )

            industry = st.selectbox(
                "Industry",
                list(mapp.industry_code_mapping.keys()),
                help="Select the cause of injury.",
            )


    # ADMINISTRATIVE INFORMATION
    with st.expander("Administrative Information"):

        col1, col2 = st.columns(2)

        with col1:
            carrier_name = st.selectbox(
                "Carrier Name",
                mapp.carrier_name,
                help="Select the insurance carrier handling the claim.",
            )

            
            carrier_type = st.selectbox(
                "Carrier Type",
                mapp.carrier_type,
                help="Select the carrier type.",
            )

            
            county = st.selectbox(
                "County of Injury",
                mapp.county_of_injury,
                help="Select the county where the injury occurred.",
            )

            district = st.selectbox(
                "District Name",
                ['SYRACUSE', 'ROCHESTER', 'ALBANY', 'HAUPPAUGE', 'NYC',
                  'BUFFALO', 'BINGHAMTON', 'STATEWIDE'
                ],
                help="Select the district of the incident.",
            )


            attorney = st.radio(
                "Claim Represented by Attorney?",
                ["Yes", "No"],
                horizontal=True,
                help="Indicate if the claim is represented by an attorney.",
            )

            alternative_dispute = st.radio(
                "Alternative Dispute Resolution?",
                ["Yes", "No"],
                horizontal=True,
                help="Indicate if alternative dispute resolution is applicable.",
            )
        with col2:

            # CONFIRMAR VALORES
            first_hearing_year = st.slider(
                "First Hearing Year",
                2020,
                2024,
                2022,
                help="Year of the first hearing.",
            )

            first_hearing_month = st.slider(
                "First Hearing Month",
                1,
                12,
                3,
                help="Month of the first hearing.",
            )

            first_hearing_day = st.slider(
                "First Hearing Day",
                1,
                31,
                12,
                help="Day of the first hearing.",
            )
            ime_4_count = st.number_input(
                "IME-4 Forms Received Count",
                min_value=0,
                max_value=100,
                value=1,
                help="Number of IME-4 forms received.",
            )

            medical_region = st.radio(
                "Medical Fee Region",
                ["I", "II", 'III', 'IV', 'X'],
                help="Number of IME-4 forms received.")



    # Collecting all inputs into the input_df dictionary
    st.subheader("Your Inputs")
    input_data = {
        'Claim Identifier': int(claim_iden),
        "Birth Year": int(birth_year),
        "Gender": str(gender),
        "Average Weekly Wage": float(avg_weekly_wage),
        "Zip Code": int(zip_code),
        "Number of Dependents": int(n_dependents),
        "Accident Date": datetime.date(accident_year, accident_month, accident_day),
        "Assembly Date": datetime.date(assembly_year, assembly_month, assembly_day),
        "C-2 Date": datetime.date(c2_year, c2_month, c2_day),
        "C-3 Date": datetime.date(c3_year, c3_month, c3_day),
        "COVID-19 Indicator": str(covid),
        "Carrier Name": str(carrier_name),
        "Carrier Type": str(carrier_type),
        "County of Injury": str(county),
        "District Name": str(district),
        "Attorney/Representative": str(attorney),
        "Alternative Dispute Resolution": str(alternative_dispute),
        "First Hearing Date": datetime.date(first_hearing_year, first_hearing_month, first_hearing_day),
        "IME-4 Count": int(ime_4_count),
        "Medical Fee Region": str(medical_region),
        "Industry Code": int(mapp.industry_code_mapping[industry]),
        "Industry Code Description": str(industry),
        "WCIO Cause of Injury Code": int(mapp.cause_of_injury_mapping[cause_injury]),
        "WCIO Nature of Injury Code": int(mapp.nature_of_injury_mapping[nature_injury]),
        "WCIO Part Of Body Code": int(mapp.part_of_body_mapping[part_of_body])
        }
    
    input_df = pd.DataFrame(input_data, index=[0])

    st.write(input_df)

    # Placeholder for prediction button
    st.subheader("Prediction")
    if st.button("Predict"):
        csv_path = "user_inputs.csv"
        input_df.to_csv(csv_path, index=False)
        st.success(f"User inputs saved to {csv_path}")

        # Call the preprocessing and prediction function
        prediction = p.preproc_(csv_path)
        st.subheader("Prediction Result")
        st.write(f"The predicted compensation benefit is: {prediction}")