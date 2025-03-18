import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pickle
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from predict import show_predict



# Load the logo
img = Image.open("./24_Nexus_Analytics.png")
logo = Image.open("./Nova_IMS.png")

#load Data for vizualizations 
df= pd.read_csv("./train_data_EDA.csv")
county_df = pd.read_csv("./geo_county.csv")

# Define the navigation menu
def streamlit_menu():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Inputs and Prediction", "Explore Data"],
        icons=["house", "check-circle", "bar-chart-line"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )
    return selected


selected = streamlit_menu()

st.logo(
    logo,
    size="large")

# Home Page
if selected == "Home":
    st.title("Grant Compensation Benefit Prediction App")

    # Dropdown for team members
    st.subheader("Project Team")
    with st.expander("Click to view team details"):
        team_members = {
            "Data Scientist Manager": ("António Oliveira", "20211595"),
            "Data Scientist Senior": ("Tomás Ribeiro", "20240526"),
            "Data Scientist Junior": ("Gonçalo Pacheco", "20240695"),
            "Data Analyst Senior": ("Gonçalo Custódio", "20211643"),
            "Data Analyst Junior": ("Ana Caleiro", "20240696"),
        }
        for role, (name, student_id) in team_members.items():
            st.write(f"**{role}**: {name} ({student_id})")

    # App description
    st.markdown("""
    This application is designed to assist the New York Workers' Compensation Board (WCB) in automating the classification of compensation claims. It uses a machine learning model trained on historical claims data from 2020 to 2022 to predict whether a compensation benefit will be granted based on various claim-related features.
    To use the application, the WCB simply needs to complete a quick form on the Inputs and Predictions page. Additionally, we provide an Explore Data page, which allows the board to analyze the data powering the model for more specific insights.
    """)

    #Inputing the metadata 
    st.subheader("Metadata")
    with st.expander("Click to view detailed information about the questions in the form"):
        metadata = {
            "County of Injury": "Name of the New York County where the injury occurred.",
            "COVID-19 Indicator": "Indication that the claim may be associated with COVID-19.",
            "District Name": "Name of the WCB district office that oversees claims for that region or area of the state.",
            "First Hearing Date": "Date the first hearing was held on a claim at a WCB hearing location. A blank date means the claim has not yet had a hearing held.",
            "Gender": "The reported gender of the injured worker.",
            "IME-4 Count": "Number of IME-4 forms received per claim. The IME-4 form is the 'Independent Examiner's Report of Independent Medical Examination' form.",
            "Industry Code": "NAICS code and descriptions are available at: https://www.naics.com/search-naics-codes-by-industry/.",
            "Industry Code Description": "2-digit NAICS industry code description used to classify businesses according to their economic activity.",
            "Medical Fee Region": "Approximate region where the injured worker would receive medical service.",
            "OIICS Nature of Injury Description": "The OIICS nature of injury codes & descriptions are available at https://www.bls.gov/iif/oiics_manual_2007.pdf.",
            "WCIO Cause of Injury Code": "The WCIO cause of injury codes & descriptions are at https://www.wcio.org/Active%20PNC/WCIO_Cause_Table.pdf.",
            "WCIO Cause of Injury Description": "See description of field above.",
            "WCIO Nature of Injury Code": "The WCIO nature of injury codes are available at https://www.wcio.org/Active%20PNC/WCIO_Nature_Table.pdf.",
            "WCIO Nature of Injury Description": "See description of field above.",
            "WCIO Part Of Body Code": "The WCIO part of body codes & descriptions are available at https://www.wcio.org/Active%20PNC/WCIO_Part_Table.pdf.",
            "WCIO Part Of Body Description": "See description of field above.",
            "Zip Code": "The reported ZIP code of the injured worker's home address.",
            "Agreement Reached": "Binary variable: Yes if there is an agreement without the involvement of the WCB -> unknown at the start of a claim.",
            "WCB Decision": "Multiclass variable: Decision of the WCB relative to the claim: 'Accident' means that claim refers to workplace accident, 'Occupational Disease' means illness from the workplace. -> requires WCB deliberation so it is unknown at start of claim.",
            "Claim Injury Type": "Main variable: Deliberation of the WCB relative to benefits awarded to the claim. Numbering indicates severity."
        }

        # Transform the Dict in a pandas data frame
        metadata_df = pd.DataFrame(list(metadata.items()), columns=["Attribute", "Description"])
        
        # Display the table
        st.dataframe(metadata_df)

    # Display the logo below the team section
    st.image(img, use_container_width=True)

#Inputs and Predictions Page 
if selected == "Inputs and Prediction":
    st.title("Predict Compensation Benefit")
    st.markdown(
        "Provide the necessary details below to predict whether a compensation benefit will be granted."
    )

    # Run the function of Predictions
    show_predict()

# Explore Data Page 
if selected == "Explore Data":
    st.title("Model Data and Insights")
    st.subheader("Analyse the pairwise relation between the numerical features")

    
    # Create the list of numeric features 
    numeric_features = ['Age at Injury', 'Average Weekly Wage', 'Birth Year', 'IME-4 Count','Industry Code','WCIO Cause of Injury Code', 'WCIO Nature of Injury Code', 'WCIO Part Of Body Code', 'Number of Dependents']
    
    # Defining and calling the function for an interactive scatterplot
    def interactive_scater (dataframe):
        x_axis_val = st.selectbox('Select X-Axis Value', options=numeric_features)
        y_axis_val = st.selectbox('Select Y-Axis Value', options=numeric_features)
        col = st.color_picker('Select a plot colour')

        plot  = px.scatter(dataframe, x=x_axis_val, y=y_axis_val)
        plot.update_traces(marker = dict(color=col))
        st.plotly_chart(plot)

    interactive_scater (df)

    st.divider()
    
    #Creating Hist of numerical
    st.subheader("Analyse the histograms of Numerical features") 

    def interactive_hist (dataframe):
        box_hist = st.selectbox('Feature', options=numeric_features)
        color_choice = st.color_picker('Select a plot colour', '#1f77b4')
        bin_count = st.slider('Select number of bins', min_value=5, max_value=100, value=20, step=1)

        hist  = sns.displot(dataframe[box_hist], color=color_choice, bins=bin_count)
        
        plt.title(f"Histogram of {box_hist}")
        st.pyplot(hist)

        
    interactive_hist(df)

    st.divider()
    
    st.subheader("Numerical features against Target Variable (Claim Injury Type)")

    def hist_target(dataframe):
         # Reverse the encoding in the target column
        label_mapping = { 
        0: "1. CANCELLED",
        1: "2. NON-COMP",
        2: "3. MED ONLY",
        3: "4. TEMPORARY",
        4: "5. PPD SCH LOSS",
        5: "6. PPD NSL",
        6: "7. PTD",
        7: "8. DEATH"   
    }
    
        # Map the numeric Claim Injury Type to descriptive labels
        dataframe['Claim Injury Type Label'] = dataframe['Claim Injury Type'].map(label_mapping)

        # Reverse the encoding in the target column
        target_var = dataframe['Claim Injury Type']
    
        # Create a dropdown for selecting numeric features
        target_hist = st.selectbox('Feature', options=numeric_features, key="numeric_feature_hist")
    
        # Create a dropdown for selecting the number of bins
        bin_count = st.slider('Select number of bins', min_value=5, max_value=100, value=30, key="bin_slider")

        color_palette = "inferno"

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(data=dataframe, x=target_hist, hue="Claim Injury Type Label", kde=True, bins=bin_count, palette=color_palette)
        plt.title(f"Distribution of {target_hist} by Claim Injury Type")
        st.pyplot(plt)
    
    hist_target(df)

    st.divider()
    
    #Creating the map of County
    st.subheader("Map of County Injury")
    st.markdown("his map visualizes the frequency of injuries reported across various New York counties. Each point on the map represents a county, hover over the points to see more details about each one of them.")
    
    
    def dysplay_map (Dataframe):

        fig = px.scatter_mapbox(Dataframe, lat ='Latitude',lon='Longitude', 
                                color="Frequency", color_continuous_scale=px.colors.sequential.Inferno, zoom=3
                                )
        
        fig.update_layout(mapbox_style='open-street-map')

        st.plotly_chart(fig)
    
    dysplay_map(county_df)
    
    
    
    

   