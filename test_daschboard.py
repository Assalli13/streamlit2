import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go
import requests
import json

st.title('"Projet 7 - Implémentez un modèle de scoring"') 

file1 = open('best_model_B_S.pkl', 'rb')
model = pickle.load(file1)
file1.close()


data = pd.read_csv("X_valid_sample_sm")
#data['SK_ID_CURR'] = data['SK_ID_CURR'].astype(int)
data = data.reset_index()
data['SK_ID_CURR'] = data['index']
data = data.drop('index', axis = 1)
init = st.markdown("*Initialisation de l'application en cours...*")
liste_id = data['SK_ID_CURR'].tolist()


y_pred = model.predict(data.drop('Unnamed: 0', axis =1))
prediction = model.predict_proba(data.drop('Unnamed: 0', axis =1))
y_pred_test = (prediction > 0.1)
y_pred_test = np.array(y_pred_test > 0) * 1
#st.write(data)
st.write(pd.DataFrame({
    'Id': data['SK_ID_CURR'],
    'pred': list(y_pred_test),
}))
data['TARGET'] = list(prediction[:, 1])

if(st.button('Predict')):

    st.write(data)
    client = data[data['SK_ID_CURR']==1]
    # Afficher les données filtrées
    st.write(client)
        # Afficher la valeur de la colonne 'TARGET' pour le client sélectionné
    score_client = client['TARGET']
    st.write(score_client)

def get_predict_of_id():
#Get the client ID from the user
    client_id = st.number_input("Enter the client ID:")

    if st.button('Get Score'):
    # data to send in the request body
        id_client = {"SK_ID_CURR": client_id}
        
        st.write('bonjour')

    # send the POST request
        response = requests.get("https://assalli13-flask-test-api-ndlz8b.streamlit.app/", json=id_client)
        st.write('bonjour')
        
        

     # get the response data as a python object
       
       
        response_data = json.loads(response.text)
        response_data = response.json()
        st.write(response_data)
        

    return client_id, response_data


#Affichage du score cilent
def gauge_chart(thres):
    client_id, response_data = get_predict_of_id()
    client_data = data.query("SK_ID_CURR == @client_id")
    st.write(client_data)
    if not client_data.empty:
        #percent_sup_seuil = 100 * np.sum(response_data['prediction_proba']> thres) / client_data.shape[0]
        st.write(response_data['prediction_proba'])
        percent_sup_seuil = 100*response_data['prediction_proba']
        if percent_sup_seuil<=100*thres:
            st.write('Demande peut etre acceptée')
        else:
            st.write('Demande risqe detre refusée')
        if not np.isnan(percent_sup_seuil):
        # Define color mapping
            if percent_sup_seuil <= 30:
                 color = 'black'
            elif percent_sup_seuil <= 60:
                color = 'black'
            else:
                color = 'black'
        
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = response_data['prediction_proba']*100,
                title = {"text": "Pourcentage de solvabilité des clients dans dataset"},
                 domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {'axis': {'range': [None, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [20, 100],'color': 'red'},
                            {'range': [10, 20], 'color': 'orange'},
                            {'range': [0, 10], 'color': 'green'}
                                                            ],
                                    'threshold': {
                          'line': {
                    'color': "black",
                     'width': 2
                              },
                     'thickness': 0.75,
                                'value': thres
                                             }
                                              }))
            st.plotly_chart(fig)
            feature_importancess = json.loads(response_data["feature_importances"])
            feature_importancess_df = pd.DataFrame(feature_importancess)
            #st.write(feature_importancess_df)
            top_4_features = feature_importancess_df.sort_values(by='importance', ascending=False)[:4]
            #st.bar_chart(top_4_features, x=top_4_features['feature'], y=top_4_features['importance'])
            fig1, ax = plt.subplots(figsize=(10, 4))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            ax.barh(top_4_features['feature'], top_4_features['importance'], color=colors)
            ax.set_xlabel('Feature')
            ax.set_title('Top 4 most important features')
            ax.invert_yaxis()

            # Add grid lines to make the chart more readable
            ax.grid(True, which='both', axis='x', linestyle='--', color='gray')

            st.pyplot(fig1, ax)
            #st.write(top_4_features)
            st.write(top_4_features)
        else:
            st.warning("No data available for this client id or the data is not valid")
    else:
        st.warning("No data available for this client id or the data is not valid")
gauge_chart(thres = 0.15)

def show_overview():
    st.title("Risque")
    risque_threshold = st.slider(label = 'Seuil de risque', min_value = 0.0,
                    max_value = 1.0 ,
                     value = 0.5,
                     step = 0.1)
    st.write(0.5)
show_overview()
st.write(data)
 
def gauge_chart(thres):
    percent_sup_seuil = 100*(data['TARGET'] > thres).sum()/data.shape[0]
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = percent_sup_seuil,
        title = {"text": "Pourcentage de solvabil des clients di dataset"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {'axis': {'range': [None, 100]},
            'bar': {'color': "red"},
            'steps' : [
                {'range': [0, 50], 'color': 'lightgray'},
                {'range': [50, 75], 'color': 'gray'},
                {'range': [75, 100], 'color': 'darkgray'}
            ],
            'threshold' : {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': thres}}))
    st.plotly_chart(fig) 
gauge_chart(thres = 0.1)

def hist_graph ():
    bins = st.slider("Number of bins",5,50,10)
    hist_values, bin_edges = np.histogram(data['DAYS_BIRTH'], bins=bins)
    bin_edges = bin_edges[:-1] # to match the number of values
    hist_df = pd.DataFrame({"bin_edges": bin_edges,"hist_values": hist_values})
    st.bar_chart(hist_df)
    df = pd.DataFrame(data[:200],columns = ['DAYS_BIRTH','AMT_CREDIT'])
    fig, ax = plt.subplots(1, 2)
    df['DAYS_BIRTH'].hist(ax=ax[0],bins=bins)
    ax[0].set_title('DAYS_BIRTH')
    df['AMT_CREDIT'].hist(ax=ax[1],bins=bins)
    ax[1].set_title('AMT_CREDIT')
    st.pyplot(fig)
hist_graph()
        
