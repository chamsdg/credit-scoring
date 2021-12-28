# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_validate
from lightgbm import LGBMClassifier
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
from my_functions.functions_cached import *  # personnal functions pkg and module
#######################################################################################
# To run this code, type in terminal at the file path: 
# streamlit run app3.py
#######################################################################################
# Stating graphical parameters
COLOR_BR_r = ['#00CC96', '#EF553B'] #['dodgerblue', 'indianred']
COLOR_BR = ['indianred', 'dodgerblue']
#######################################################################################
# Managing data import/export
PATH = "C:/Users/BY5/CHAMSEDINE_CREDIT_SCORING/deploiement/" # os.getcwd()+'/' # change for online app or local
PATH_INPUT = PATH+"input/"
FILENAME_TRAIN = PATH_INPUT+'application_train_sample.csv' # sample of train set for online version 25MB
FILENAME_TEST = PATH_INPUT+'application_test.csv'
FILENAME_MODEL = PATH+'optimized_model.sav'

#######################################################################################
# Setting layout & navigation pane
st.set_page_config(page_title="Baamtu Credit Score Dashboard", # Must be 1st st statement
                   page_icon="☮",
                   initial_sidebar_state="expanded")

df_train = get_data(FILENAME_TRAIN) # load trainset data in a df
df_test = get_data(FILENAME_TEST) # load testset (unlabeled) data in a df

sb = st.sidebar # add a side bar 
sb.image('https://baamtu.com/wp-content/uploads/2018/07/logo_baamtu_web.png', width=300)
sb.image('https://baamtu.com/wp-content/uploads/2019/11/FORMATION.png', width=300)
rad_who = sb.radio('', ['🤵 Bank Clerk']) # two versions of the app
# the two versions of the app will have different options, home is common to all
if rad_who == '🤵 Bank Clerk':
    sb.markdown('**Client to scout:**')
    np.random.seed(13) # one major change is that client is directly asked as input since sidebar
    label_test = df_test['SK_ID_CURR'].sample(50).sort_values()
    radio = sb.radio('', ['Random client ID', 'Type client ID'])
    if radio == 'Random client ID': # Choice choose preselected seed13 or a known client ID
        input_client = sb.selectbox('Select random client ID', label_test)
    if radio == 'Type client ID':
        input_client = int(sb.text_input('Type client ID', value=147254))

    sb.markdown('**Navigation**')
    rad = sb.radio('', ['🏠 Home', 
    '🔎 Client data',
    '👁️ Deep Short Analysis', 
    '📉 Client prediction'])
else:
    sb.markdown('**Navigation**')
    rad = sb.radio('', ['🏠 Home'])

# defining containers of the app
header = st.container() 
dataset = st.container()
eda = st.container()
model_training = st.container()
model_predict = st.container()

#######################################################################################
# Implementing containers
#######################################################################################

if rad == '🏠 Home': # with this we choose which container to display on the screen
    with header:
        a,z,e = st.columns(3) #OOP style 
        a.image('https://baamtu.com/wp-content/uploads/2018/07/mediciel-600x450.jpg', width=60)
        z.image('https://baamtu.com/wp-content/uploads/2018/07/erp-600x450.jpg', width=60)
        e.image('https://baamtu.com/wp-content/uploads/2018/07/leboncompte-600x450.jpg', width=50)

        st.title("Baamtu Credit Score Dashboard! \n ----")
        st.header("**AIDARA CHAMSEDINE **")
        st.markdown("L'objectif de ce projet :")
        st.markdown("Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon     automatique. Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les     prédictions faites par le modèle et d’améliorer la connaissance client des chargés de relation client.")
        st.markdown("Let's start")
        st.markdown("- A client-oriented version 🤵")


#######################################################################################

if rad == '👁️ Deep Short Analysis':
    with dataset:
        st.header("**Short Analysis.** \n ----") # title > header > subheader > markdown ~ text
        st.markdown("In this project, we focus only on the application train dataset.")
        
        st.subheader("Here's the dataframe.")
        max_row = st.slider("Select at many row you wanna visualize", value=1000, min_value=1, max_value=len(df_train)) 
        st.write(df_train.head(max_row))
        
        #st.subheader("Heatmap with missing data.")
        #st.markdown('Showing records in dark, missing values in light. Numeric values will be subsequently imputed with median for model training.')

        #st.plotly_chart(heatmap(df_train, max_row)) # heatmap is a home-made func 

        val_notnull = df_train.count().sum()
        lig,col = df_train.shape
        taux = val_notnull/(lig*col)*100  
        # Informations générales pour le diagramme circulaire 
        fig, ax = plt.subplots(figsize=(5, 5))
        st.title("Taux de remplissage") 
        ax.axis("equal") 
        # Ploter la figure pie
        ax.pie([taux, 100 - taux], labels=["Taux de remplissage", "Taux de valeurs manquantes"],autopct='%1.2f%%',explode=(0,0.1),radius=1)
        #st.write(["Taux de remplissage", "Taux de valeurs manquantes"])
        st.pyplot(fig)
        # from my_functions.cached_funtions, it's important to cache functions to save loading times

#######################################################################################           

#######################################################################################

if rad == '💪 Model training': 
    with model_training:
        st.header("**Model training.** \n ----")
        st.markdown("We'll be using LightGBM Classifier (Microsoft),\
            as state-of-the-art gradient boosting classifier.\
                \n You can tune  hyperparameters, fit and observe\
                cross-validation scores (using 3 folds).")

        _, col2, _ = st.columns(3)
        col2.image('https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/logo/LightGBM_logo_black_text_tiny.png')     
        # preprocess = home-made func, with 3 outputs (X_train_sc, X_test_sc, feat_list)
        X_train_sc, _, _ = preprocess(df_train, df_test)
        y_train = df_train['TARGET']
        
        col1, col2 = st.columns(2)
        col1.subheader("**Tuning best hyperparameters.**")
        # sliders for hyperprams of LightGBM classifier
        n_estimators = col1.slider("Number of trees", value=300, min_value=200, max_value=1000)
        num_leaves = col1.slider("Number of leaves", value=10, min_value=5, max_value=100)
        lr = col1.select_slider("Learning rate", options=[1e-4, 1e-3, 1e-2, 1e-1, 1e0], value=1e-1)
        scale_pos_weight = col1.select_slider("Weight of positives (>10 highly recommanded)",\
            options=[1e-1, 1e0, 1e1, 2e1, 5e1, 1e2], value=1e1) # as alternative for log sliders
        reg_alpha = col1.slider("L1 regularization term", value=0, min_value=0, max_value=100)
        reg_lambda = col1.slider("L2 regularization term", value=0,  min_value=0, max_value=100)
        checkbox = col1.checkbox("Export optimized model 🥒🥒🥒") # export or not model checkbox

        if col1.button('Fit using cross-validation!'):
            col2.subheader('**Validation set fit scores.**')
            st.spinner('Fitting...') # not working...
            model = LGBMClassifier(max_depth=-1,
                                    random_state=13,
                                    silent=True,
                                    metric='none',
                                    n_jobs=-1,
                                    n_estimators=n_estimators,
                                    num_leaves=num_leaves,
                                    learning_rate=lr,
                                    scale_pos_weight=scale_pos_weight,
                                    reg_alpha=reg_alpha,
                                    reg_lambda=reg_lambda
                                )

            scoring = ['roc_auc','precision','recall','f1']
            x_val = cross_validate(model, X_train_sc, y_train, cv=3, scoring=scoring)
            # putting output of Xval for easier aggregations
            time, unk, auc, precision, recall, f1 = pd.DataFrame(x_val).mean(axis=0)
            d_time, d_unk, d_auc, d_precision, d_recall, d_f1 = pd.DataFrame(x_val).std(axis=0)

            col2.subheader('Mean fit time (s)')
            col2.write(f'{time:.0f} ± {d_time:.0f}')
            col2.subheader('AUC-score')
            col2.write(f'{auc:.0%} ± {d_auc:.0%}')
            col2.subheader('Recall')
            col2.write(f'{recall:.0%} ± {d_recall:.0%}')
            col2.subheader('Precision')
            col2.write(f'{precision:.0%} ± {d_precision:.0%}')
            col2.subheader('f1-score')
            col2.write(f'{f1:.0%} ± {d_f1:.0%}')

            if checkbox: # export with pickle
                model.fit(X_train_sc, y_train)
                pickle.dump(model, open(FILENAME_MODEL, 'wb'))
                st.header('**Successful model export!**')
                st.balloons()
                
                


#######################################################################################
if rad == '🔎 Client data': 
    with eda:
        st.header("**Client's data.** \n ----")
        # retrieving whole row of client from sidebar input ID
        client_data = df_test[df_test.SK_ID_CURR == input_client]
        client_data = client_data.dropna(axis=1) # avoiding bugs

        st.subheader(f"**Client ID: {input_client}.**")
        # plotting features from train set, with client's data as dashed line (client!=None in func)
        st.subheader("Ranking client in some features.")      
        col1, col2, col3 = st.columns(3)
        col1.plotly_chart(histogram(df_train, x='CODE_GENDER', client=[df_test, input_client]), use_container_width=True)
        col2.plotly_chart(histogram(df_train, x='EXT_SOURCE_1', client=[df_test, input_client]), use_container_width=True)
        col3.plotly_chart(histogram(df_train, x='EXT_SOURCE_2', client=[df_test, input_client]), use_container_width=True)

        st.subheader("Let's plot some extra ranking for numerical features.")
        col1, col2, col3 = st.columns(3)
        num_col = client_data.select_dtypes(include=np.number).columns.sort_values()
        input1 = col1.selectbox('1st plot', num_col)
        input2 = col2.selectbox('2nd plot', num_col[1:])
        input3 = col3.selectbox('3rd plot', num_col[2:])

        st.subheader("Now, you may pick some categorical features to plot.")
        col4, col5, col6 = st.columns(3)
        cat_col = client_data.select_dtypes(exclude=np.number).columns.sort_values()
        input4 = col4.selectbox('1st plot', cat_col[1:])
        input5 = col5.selectbox('2nd plot', cat_col[2:])
        input6 = col6.selectbox('3rd plot', cat_col[3:])

        button = st.button('Plot it! ')
        if button:
            col1.plotly_chart(histogram(df_train, x=input1, legend=False, client=[df_test, input_client]),use_container_width=True)
            col2.plotly_chart(histogram(df_train, x=input2, legend=False, client=[df_test, input_client]),use_container_width=True)
            col3.plotly_chart(histogram(df_train, x=input3, legend=False, client=[df_test, input_client]),use_container_width=True)
            col4.plotly_chart(histogram(df_train, x=input4, legend=False, client=[df_test, input_client]),use_container_width=True)
            col5.plotly_chart(histogram(df_train, x=input5, legend=False, client=[df_test, input_client]),use_container_width=True)
            col6.plotly_chart(histogram(df_train, x=input6, legend=False, client=[df_test, input_client]),use_container_width=True)
        
        st.subheader("More information about this client.")
        # displaying values from a dropdown (had issues with NaNs that's why .dropna())
        col1, col2 = st.columns(2)
        info = col1.selectbox('What info?', client_data.columns.sort_values())     
        info_print = client_data[info].to_numpy()[0]

        col1.subheader(info_print)
        # displaying whole non NaNs row
        col2.write("All client's data.")
        col2.write(client_data)

#######################################################################################



if rad == '📉 Client prediction': 
    with model_predict:
        st.header("**Client solvency prediction.** \n ----")

        col1, col2 = st.columns(2)
        col1.markdown(f'**Client ID: {input_client}**')

        if col2.button('Predict & plot!'):
            # this time we need all outputs of preprocessing                    
            X_train_sc, X_test_sc, feat_list = preprocess(df_train, df_test)
            y_train = df_train['TARGET']
            # calling pretrained model from pickle file (.sav)
            try: 
                model = pickle.load(open(FILENAME_MODEL, 'rb'))
            except:
                raise 'You must train the model first.'
            # finding client row index in testset
            idx = df_test.SK_ID_CURR[df_test.SK_ID_CURR == input_client].index
            client = X_test_sc[idx, :] # for then slicing preprocessed test data
            
            y_prob = model.predict_proba(client) # predicting proba
            y_prob = [y_prob.flatten()[0], y_prob.flatten()[1]] #misalignement of format
            # importance of features extracted using scikit learn: pred_contrib=True
            imp_feat = model.predict_proba(X_test_sc[idx, :], pred_contrib=True).flatten()
            imp = pd.DataFrame([feat_list, imp_feat]).T.sort_values(by=1, ascending=False).head(20)

            col1, col2 = st.columns(2)
            # adapting message wether client's pos or neg
            if y_prob[1] < y_prob[0]:
                col1.subheader(f"**Successful payment probability.**")
            else:
                col1.subheader(f"**Failure payment probability.**")
            # plotting pie plot for proba, finding good h x w was a bit tough
            fig = px.pie(values=y_prob, names=[0,1], color=[0,1], color_discrete_sequence=COLOR_BR_r, 
            width=230, height=230)
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            col1.plotly_chart(fig, use_container_width=True)

            col2.subheader("**Client spiderchart.**")
            # plotting radar chart
            columns = (imp.head(5)[0].values) # recovering top5 most important features as... tuples, why did I do that???
            df_test_sc = pd.DataFrame(X_test_sc, columns=feat_list)
            # I wanted to plot average that's why I made a df, but I think it's useless now 
            # since it was a bit difficult and I drop this idea. Instead, I kept scaled version of data
            # so average should be zero and 1 = +1 sigma (StandardScaler)
            client_radar = df_test_sc.loc[idx,columns].T.reset_index()
            client_radar = client_radar.rename(columns={"index":"theta", idx.values[0] :'r'})

            fig = px.line_polar(client_radar, 
                                theta='theta', 
                                r='r', 
                                log_r=False, 
                                line_close=True,
                                color_discrete_sequence=['indianred'],
                                width=250,
                                height=250,
                                )
            fig.update_traces(fill='toself')
            fig.update_layout(margin=dict(l=50, r=50, t=50, b=10))  
            col2.plotly_chart(fig, use_container_width=True)

            st.subheader("**Importance of features to decision.**")
            # then plotting feature importance, but for readibility slicing absissa labels using:
            labels = [(i[:7] + '...'+i[-7:]) if len(i) > 17 else i for i in imp[0]]
            fig = px.bar(   imp.head(10),
                            x=0,
                            y=1,
                            width=300,
                            height=300,
                            color=range(10),
                            color_continuous_scale='OrRd_r',
                            orientation='v')
            fig.update(layout_coloraxis_showscale=False)
            fig.update_xaxes(title='')
            fig.update_layout(xaxis = dict(
                            tickmode = 'array',
                            tickvals = [i for i in range(20)],
                            ticktext = labels))
            fig.update_yaxes(title='Relative importance')
            fig.update_yaxes(showticklabels=False)
            fig.update_layout(margin=dict(l=20, r=20, t=10, b=10))                
            st.plotly_chart(fig, use_container_width=True)

            # one-hot-encoded columns added a "_string" as lower case to col names
            # thus checking if the col name is full upper case if a good test to 
            # check whether the col is num or cat (I want only 6 most num feats here)
            num_plots=[]
            i=0
            while (i in range(len(imp))) and (len(num_plots) < 7):
                if imp.iloc[i,0] == imp.iloc[i,0].upper():
                    num_plots.append(imp.iloc[i,0])
                i+=1

            st.subheader("Ranking client in some important features.")      
            col1, col2, col3 = st.columns(3)
            col1.plotly_chart(histogram(df_train, x=num_plots[0], client=[df_test, input_client]), use_container_width=True)
            col2.plotly_chart(histogram(df_train, x=num_plots[1], client=[df_test, input_client]), use_container_width=True)
            col3.plotly_chart(histogram(df_train, x=num_plots[2], client=[df_test, input_client]), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.plotly_chart(histogram(df_train, x=num_plots[3], client=[df_test, input_client]), use_container_width=True)
            col2.plotly_chart(histogram(df_train, x=num_plots[4], client=[df_test, input_client]), use_container_width=True)
            col3.plotly_chart(histogram(df_train, x=num_plots[5], client=[df_test, input_client]), use_container_width=True)

#######################################################################################
if __name__ == "__main__":
    print("Script runned directly")
else:
    print("Script called by other")