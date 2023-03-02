import streamlit as st
import pandas as pd
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.express as px
import requests
import json

st.title('"Projet 7 - Implémentez un modèle de scoring"') 

file1 = open('final_model1.pkl', 'rb')
model = pickle.load(file1)
file1.close()


data = pd.read_csv("X_train_2")
data_test1 = pd.read_csv("X_test_2")
#data_test = data_test1.drop(['Unnamed: 0', 'TARGET'], axis =1)
#st.write(data_test1)
init = st.markdown("Réalisé par: Mohamed Assali")
#liste_id = data['SK_ID_CURR'].tolist()
#--------------------------------------------------------------------------------------------------------------------------------------------#
data_test = data_test1.drop(['Unnamed: 0', 'TARGET'], axis =1)
prediction_test = model.predict_proba(data_test)
y_pred_data_test = (prediction_test > 0.1)
y_pred_data_test = np.array(y_pred_data_test > 0) * 1
data_test['TARGET'] = list(1-prediction_test[:, 0])
#st.write(data_test)
#--------------------------------------------------------------------------------------------------------------------------------------------#
y_pred = model.predict(data.drop('Unnamed: 0', axis =1))
prediction = model.predict_proba(data.drop('Unnamed: 0', axis =1))
y_pred_test = (prediction > 0.1)
y_pred_test = np.array(y_pred_test > 0) * 1
#st.write(data_test1)
#st.write(pd.DataFrame({
    #'Id': data['SK_ID_CURR'],
    #'pred': list(y_pred_test),
#}))
data['TARGET'] = list(prediction[:, 1])

if(st.button('Get info')):

    st.write(data_test1)
    
    def pie_chart(thres):
        percent_sup_seuil = 100 * (data_test['TARGET'] > thres).sum() / data_test.shape[0]
        percent_inf_seuil = 100 - percent_sup_seuil
        d = {
         'col1': [percent_sup_seuil, percent_inf_seuil],
        'col2': ['% Non Solvable', '% Solvable']}
        df = pd.DataFrame(data=d)
        fig = px.pie(df, values='col1', names='col2', title='Pourcentage de solvabilité des clients dans le dataset')
    
        fig.update_traces(textposition='inside', texttemplate='%{percent:.2f}%',
            marker=dict(
                colors=['#ff7f0e', '#1f77b4'],
                line=dict(color='#FFFFFF', width=2)))

        fig.update_layout(
            title={'text': 'Répartition des clients selon la solvabilité', 'font': dict(size=24),'x': 0.5,'xanchor': 'center'},
            legend=dict(orientation='h',yanchor='bottom', y=1.02,
            xanchor='right', x=1), margin=dict(l=20, r=20, t=80, b=20), paper_bgcolor='#F2F2F2', font=dict(size=14, color='#4d4d4d'))
    
        st.plotly_chart(fig)
    pie_chart(thres = 0.15)
    
AMT_credit = st.sidebar.number_input('AMT_CREDIT', -2.0, 7.0, step=0.0001, format="%.4f") 
income = st.sidebar.number_input('AMT_ANNUITY',  -2.0, 10.0,step=0.00001, format="%.4f")
age = st.sidebar.number_input('DAYS_BIRTH', -4.0, 4.0, step=0.0001, format="%.4f")
ext_source_1 = st.sidebar.number_input('EXT_SOURCE_1', -4.0, 4.0, step=0.0001, format="%.4f")
ext_source_2 = st.sidebar.number_input('EXT_SOURCE_2', -4.0, 4.0,step=0.0001, format="%.4f")
ext_source_3 = st.sidebar.number_input('EXT_SOURCE_3', -4.0, 4.0, step=0.0001, format="%.4f")

if st.button('Get Score'):
    def get_predict_of_id():
    #Get the client ID from the user

        data = {
            'AMT_ANNUITY': income,
            "AMT_CREDIT": AMT_credit,
            'DAYS_BIRTH': age,
            'EXT_SOURCE_1': income,
            'EXT_SOURCE_2': ext_source_2,
            'EXT_SOURCE_3': ext_source_3}
        
        
        #response = requests.post("http://localhost:5000/predictByClientId", json=data)
        response = requests.post("https://flask-1.assalli13.repl.co/predictByClientId", json=id_client)

        if response:
            
         
            #response = requests.post("http://assali.pythonanywhere.com//predictByClientId", json=id_client)

         # get the response data as a python object
            response_data = json.loads(response.text)
            response_data = response.json()
            #st.write(response_data)
        else:
             print('NO')

        return data, response_data
    def show_overview():
            st.title("Risque")
            risque_threshold = st.slider(label = 'Seuil de risque', min_value = 0.0,
                            max_value = 1.0 ,
                             value = 0.1,
                             step = 0.1)
            st.write(0.1)
    show_overview()


    #Affichage du score cilent
    def gauge_chart(thres):
        #client_id, response_data = get_predict_of_id()
        data, response_data = get_predict_of_id()
        
        client_data = pd.DataFrame(data, index=pd.Index([0]))
        #client_data = data.query("SK_ID_CURR == @client_id")
        st.write(client_data)
        if not client_data.empty:
            percent_sup_seuil = 100 * np.sum(response_data['prediction_proba']> thres) / client_data.shape[0]
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

    
    #st.write(data)
     
    def gauge_chart(thres):
        percent_sup_seuil = 100*(data['TARGET'] > thres).sum()/data.shape[0]
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = percent_sup_seuil,
            title = {"text": "Pourcentage de solvabilité des clients dans le dataset_test"},
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
    gauge_chart(thres = 0.15)

    def hist_graph ():
        bins = st.slider("Number of bins",5,50,10)
        hist_values, bin_edges = np.histogram(data['EXT_SOURCE_1'], bins=bins)
        bin_edges = bin_edges[:-1] # to match the number of values
        hist_df = pd.DataFrame({"bin_edges": bin_edges,"hist_values": hist_values})
        st.bar_chart(hist_df)
        df = pd.DataFrame(data[:200],columns = ['EXT_SOURCE_1','AMT_CREDIT'])
        fig, ax = plt.subplots(1, 2)
        df['EXT_SOURCE_1'].hist(ax=ax[0],bins=bins)
        ax[0].set_title('EXT_SOURCE_1')
        df['AMT_CREDIT'].hist(ax=ax[1],bins=bins)
        ax[1].set_title('AMT_CREDIT')
        st.pyplot(fig)
    hist_graph()

def bivariate_analysis(data, var_selection):
    if len(var_selection) < 2:
        st.warning("Sélectionnez au moins deux variables pour l'analyse bivariée.")
        return

    # Calculer la matrice de corrélation entre les variables sélectionnées
    corr = data[var_selection].corr()

    # Créer la heatmap de la matrice de corrélation
    fig = px.imshow(corr.values, x=corr.index, y=corr.columns, color_continuous_scale='RdBu')

    # Spécifier le titre de la barre de couleur
    fig.update_layout(coloraxis_colorbar_title='Corrélation')
    

    # Afficher la heatmap avec Streamlit
    st.plotly_chart(fig)

# Sélection des variables à analyser
var_selection = st.multiselect('Sélectionnez les variables à analyser', data_test1.columns)

# Bouton pour lancer l'analyse bivariée
if st.button('Analyse bivariée'):
    bivariate_analysis(data_test1, var_selection)



        

     



        

     


