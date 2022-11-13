from sklearn import model_selection, metrics, ensemble
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import torch


st.set_page_config(layout = "wide")
use_cuda = torch.cuda.is_available()

st.title('Accenture - Supply Chain Resilience Challenge')

st.write("## 1- Global visualisation")
uploaded_file = st.file_uploader("Choose dataset with coordinate :", accept_multiple_files=False)
if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file, sep=',')
    df1 = df1.drop(['Unnamed: 0'], axis =1)
    st.write("### A- First 10 rows of the dataframe :")
    st.write(df1.head(10))


    #Total number of order
    size_df = len(df1.axes[0])
    st.write("### B- Number of order :", size_df)


    #Map : Logistic hub
    st.write("### C- Location of the logistic hubs :")
    df_map_hub = df1.filter(['city_from_coord_lon','city_from_coord_lat'], axis = 1)
    df_map_hub = df_map_hub.rename(columns= {'city_from_coord_lon':'lat','city_from_coord_lat': 'lon'})
    df_map_hub = df_map_hub.dropna()
    df_map_hub = df_map_hub.astype(int)
    st.map(df_map_hub)


    #Bar chart : True or False
    st.write("### D- Repartition of delays through all orders : True -> late; False -> Not late")
    df_late =  df1['late_order'].value_counts()
    st.bar_chart(df_late)


    st.write("## 2- Detailed analysis")
    opt = st.radio("Wich order do you want to analyze ?", ('Delayed', 'Not delayed'))


    if opt == 'Delayed':
        mask = df1[df1['late_order'] == True ]
        df_filtered = mask

        #Bar chart : Product most often delivered late
        df_product = df_filtered['product_id'].value_counts()
        df_product_filtered = df_product.loc[lambda x : x > 50]
        st.write("### A- Product most often delivered late :")
        st.bar_chart(df_product_filtered)


        #Bar chart : Hub with the most delays
        st.write("### B- Logistic hubs performance :")
        df_hub = df_filtered['logistic_hub'].value_counts()
        st.bar_chart(df_hub)


        #Average distance for a delivery
        df_average_dist = df_filtered.filter(['distance'], axis=1)
        df_average_dist = df_average_dist.dropna()
        avg_dist = (df_average_dist.sum())/len(df_average_dist)
        avg_dist = float(avg_dist)
        avg_dist = round(avg_dist, 2)
        st.write("### C- Average distance for shipment : ", avg_dist, " km.")


        #Custom procedure
        st.write("### D- Repartition of the customs procedures for all orders :")
        df_cust_proc = df_filtered['customs_procedures'].value_counts()
        index = df_cust_proc.index
        fig1, ax1 = plt.subplots()
        ax1.pie(df_cust_proc,labels=index, autopct='%1.1f%%', shadow=True, startangle=90)
        st.pyplot(fig1)


        #3pl
        st.write("### E- Repartition of the third-party logistic :")
        df_3pl = df_filtered['3pl'].value_counts()
        st.bar_chart(df_3pl)


        #Number unit
        df_nb_u = df_filtered.filter(['units'], axis=1)
        df_nb_u = df_nb_u.dropna()
        avg_nb_u = (df_nb_u.sum())/len(df_nb_u)
        avg_nb_u = float(avg_nb_u)
        avg_nb_u = round(avg_nb_u, 2)
        st.write("### F- Average number of units per order : ", avg_nb_u, " .")

    else:
        mask = df1[df1['late_order'] == False ]
        df_filtered = mask

        #Bar chart : Product most often delivered late
        df_product = df_filtered['product_id'].value_counts()
        df_product_filtered = df_product.loc[lambda x : x > 200]
        st.write("### A- Product most often delivered late :")
        st.bar_chart(df_product_filtered)


        #Bar chart : Hub with the most delays
        st.write("### B- Logistic hubs performance :")
        df_hub = df_filtered['logistic_hub'].value_counts()
        st.bar_chart(df_hub)


        #Average distance for a delivery
        df_average_dist = df_filtered.filter(['distance'], axis=1)
        df_average_dist = df_average_dist.dropna()
        avg_dist = (df_average_dist.sum())/len(df_average_dist)
        avg_dist = float(avg_dist)
        avg_dist = round(avg_dist, 2)
        st.write("### C- Average distance for shipment : ", avg_dist, " km")


        #Custom procedure
        st.write("### D- Repartition of the customs procedures for all orders :")
        df_cust_proc = df_filtered['customs_procedures'].value_counts()
        index = df_cust_proc.index
        fig1, ax1 = plt.subplots()
        ax1.pie(df_cust_proc,labels=index, autopct='%1.1f%%', shadow=True, startangle=90)
        st.pyplot(fig1)


        #3pl
        st.write("### E- Repartition of the third-party logistic :")
        df_3pl = df_filtered['3pl'].value_counts()
        st.bar_chart(df_3pl)


        #Number unit
        df_nb_u = df_filtered.filter(['units'], axis=1)
        df_nb_u = df_nb_u.dropna()
        avg_nb_u = (df_nb_u.sum())/len(df_nb_u)
        avg_nb_u = float(avg_nb_u)
        avg_nb_u = round(avg_nb_u, 2)
        st.write("### F- Average number of units per order : ", avg_nb_u, " .")

    see_prediction = st.checkbox('See prediction model ?')
    if see_prediction:
        st.write("## 3- Prediction model")
        #uploaded_file1 = st.file_uploader("Choose dataset for model : ", accept_multiple_files=False)
        #if uploaded_file1 is not None:

        # products and cities
        df_products = pd.read_csv("product_attributes.csv", sep=",")

        df_cities = pd.read_csv("cities_data.csv", sep=";")
        df_cities = df_cities[['city_from_name', 'city_to_name', 'distance']]
        df_cities = df_cities.drop_duplicates()
        df_cities2 = df_cities.copy()
        df_cities2['city_from_name'] = df_cities['city_to_name']
        df_cities2['city_to_name'] = df_cities['city_from_name']
        df_cities = pd.concat([df_cities, df_cities2])

        # read train data
        df_orders = pd.read_csv("orders.csv", sep=";")
        df_orders['origin_port'] = df_orders['origin_port'].replace(['ATHENAS'], 'Athens')
        df_orders['origin_port'] = df_orders['origin_port'].replace(['BCN'], 'Barcelona')
        df_orders['logistic_hub'] = df_orders['logistic_hub'].fillna('Nohub')

        df = df_orders.merge(df_products, how='left', left_on='product_id', right_on='product_id', sort=True)
        df['weight'].fillna(df['weight'].median(), inplace = True)
        df['material_handling'].fillna(6, inplace=True)

        df = df.merge(df_cities, how='left', left_on=('origin_port', 'logistic_hub'), right_on=('city_from_name', 'city_to_name'), sort=True)
        df.rename(columns={'distance': 'distance_port_hub'}, inplace=True)
        df = df.drop(['city_from_name', 'city_to_name'], axis=1)

        df = df.merge(df_cities, how='left', left_on=('logistic_hub', 'customer'), right_on=('city_from_name', 'city_to_name'), sort=True)
        df.rename(columns={'distance': 'distance_hub_customer'}, inplace=True)
        df = df.drop(['city_from_name', 'city_to_name'], axis=1)

        df = df.merge(df_cities, how='left', left_on=('origin_port', 'customer'), right_on=('city_from_name', 'city_to_name'), sort=True)
        df = df.drop(['city_from_name', 'city_to_name'], axis=1)
        df['distance_port_customer'] = np.where(df['logistic_hub'] == 'Nohub', df['distance'], 0)

        df['distance_port_hub'].fillna(0, inplace=True)
        df['distance_hub_customer'].fillna(0, inplace=True)
        df['distance_port_customer'].fillna(0, inplace=True)

        df['distance_port_hub'] = np.where(np.logical_and(df['distance_port_hub'] == 0, df['distance_port_customer'] == 0), df['distance_port_hub'].median(), df['distance_port_hub'])

        df['distance_hub_customer'] = np.where(np.logical_and(df['distance_hub_customer'] == 0, df['distance_port_customer'] == 0), df['distance_hub_customer'].median(), df['distance_hub_customer'])

        df = df.drop(['distance', 'order_id', 'product_id'], axis=1)
        df['total_weight'] = df['weight'] * df['units']

        #df = df.convert_dtypes()

        # read train data
        df_test = pd.read_csv("test.csv", sep=";")
        df_test['origin_port'] = df_test['origin_port'].replace(['ATHENAS'], 'Athens')
        df_test['origin_port'] = df_test['origin_port'].replace(['BCN'], 'Barcelona')
        df_test['logistic_hub'] = df_test['logistic_hub'].fillna('Nohub')

        df_test = df_test.merge(df_products, how='left', left_on='product_id', right_on='product_id', sort=True)
        df_test['weight'].fillna(df_test['weight'].median(), inplace = True)
        df_test['material_handling'].fillna(6, inplace=True)

        df_test = df_test.merge(df_cities, how='left', left_on=('origin_port', 'logistic_hub'), right_on=('city_from_name', 'city_to_name'), sort=True)
        df_test.rename(columns={'distance': 'distance_port_hub'}, inplace=True)
        df_test = df_test.drop(['city_from_name', 'city_to_name'], axis=1)

        df_test = df_test.merge(df_cities, how='left', left_on=('logistic_hub', 'customer'), right_on=('city_from_name', 'city_to_name'), sort=True)
        df_test.rename(columns={'distance': 'distance_hub_customer'}, inplace=True)
        df_test = df_test.drop(['city_from_name', 'city_to_name'], axis=1)

        df_test = df_test.merge(df_cities, how='left', left_on=('origin_port', 'customer'), right_on=('city_from_name', 'city_to_name'), sort=True)
        df_test = df_test.drop(['city_from_name', 'city_to_name'], axis=1)
        df_test['distance_port_customer'] = np.where(df_test['logistic_hub'] == 'Nohub', df_test['distance'], 0)

        df_test['distance_port_hub'].fillna(0, inplace=True)
        df_test['distance_hub_customer'].fillna(0, inplace=True)
        df_test['distance_port_customer'].fillna(0, inplace=True)

        df_test['distance_port_hub'] = np.where(np.logical_and(df_test['distance_port_hub'] == 0, df_test['distance_port_customer'] == 0), df_test['distance_port_hub'].median(), df_test['distance_port_hub'])

        df_test['distance_hub_customer'] = np.where(np.logical_and(df_test['distance_hub_customer'] == 0, df_test['distance_port_customer'] == 0), df_test['distance_hub_customer'].median(), df_test['distance_hub_customer'])

        df_test = df_test.drop(['distance', 'product_id'], axis=1)
        df_test['total_weight'] = df_test['weight'] * df_test['units']

        df_test = df_test.convert_dtypes()

        columns_to_dummies = ['origin_port','logistic_hub', 'customer', '3pl', 'customs_procedures']
        df = pd.get_dummies(df, columns=columns_to_dummies)
        df_test = pd.get_dummies(df_test, columns=columns_to_dummies)

        # split data
        X_train, X_validation, y_train, y_validation = model_selection.train_test_split(df.drop('late_order', axis=1), df['late_order'], test_size=0.20, random_state=79)
        X_test = df_test

        # training model
        classifier = ensemble.RandomForestClassifier(n_estimators=50, max_features="auto", random_state=79)

        random_grid = {#'n_estimators' : [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)], 
                        'max_features' : ['auto', 'sqrt'], 
                        'max_depth' : [int(x) for x in np.linspace(30, 90, num = 10)], 
                        'min_samples_split' : [10, 15, 20, 25, 30], 
                        'min_samples_leaf' : [2, 3, 4, 5, 6], 
                        'bootstrap' : [True, False]}

        classifier_random = model_selection.RandomizedSearchCV(estimator = classifier, param_distributions=random_grid, n_iter=60, cv=3, verbose=10, random_state=79, n_jobs=-1) 
        classifier_random.fit(X_train, y_train)
        best_params = classifier_random.best_params_

        best_forest = ensemble.RandomForestClassifier(**best_params)
        best_forest.fit(X_train, y_train)
        pred = best_forest.predict(X_validation)
        metrics.accuracy_score(y_validation, pred)

        param_grid = { 
            'max_features': ['sqrt'], 
            'max_depth': [58],
            'min_samples_split' : [int(x) for x in np.linspace(10, 30, num = 15)],
            'min_samples_leaf' : [4], 
            'bootstrap': [True], 
        }
        rf = ensemble.RandomForestRegressor()
        grid_search = model_selection.GridSearchCV(estimator = rf, param_grid = param_grid, cv=3, verbose=10, n_jobs=-1)

        grid_search.fit(X_train, y_train)
        best_grid = grid_search.best_estimator_

        classifier = ensemble.RandomForestClassifier(n_estimators=1000, max_depth=58, max_features='sqrt', min_samples_leaf=4,
                            min_samples_split=21, bootstrap=True, random_state=79)
        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_validation)
        metrics.accuracy_score(y_validation, pred)

        pred_proba = classifier.predict_proba(X_test.drop('order_id', axis=1))
        submission = pd.DataFrame({"order_id": X_test.order_id, "late_order": pred_proba[:,1]})
        st.write(submission)
        #submission.to_csv("submission_kaggle.csv", index=False)

        #n = [50, 100, 200, 500, 1000, 1500, 2000]
        #for i in range(7):
            #classifier = ensemble.RandomForestClassifier(n_estimators=n[i], max_depth=35, max_features='sqrt', min_samples_leaf=2, min_samples_split=42, bootstrap=True, random_state=79)
            #classifier.fit(X_train, y_train)
            #pred_proba = classifier.predict_proba(X_test.drop('order_id', axis=1))
            #submission = pd.DataFrame({"order_id": X_test.order_id, "late_order": pred_proba[:,1]})
            #submission.to_csv("submission_kaggle.csv" + str(i), index=False)