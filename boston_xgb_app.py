import streamlit as st, pandas as pd, numpy as np, xgboost as xgb, pickle, matplotlib, matplotlib.pyplot as pl, shap, altair as alt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap
matplotlib.use('Agg')

def main():
    st.title('Assessing Best Model Features on the Boston Housing Set')
    st.subheader('Created by: Stephan de Goede')
    st.subheader('This website will go offline on 11-28, please check my Github (https:// github.com/sgoede/streamlit-boston-app to check for a new deployed version of the app')
    @st.cache
    def load_data():
        boston = load_boston()
        return boston
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    # Load 10,000 rows of data into the dataframe.
    boston = load_data()
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')
    st.write('The detailed description of the data is listed below')
    st.write(boston.DESCR)
    st.title('Creating a Pandas Dataframe from the original dataset')
    @st.cache
    def load_dataframe():
        Boston = pd.DataFrame(boston.data, columns=boston.feature_names)
        Boston['MEDV'] = boston.target
        return Boston
    # Create a text element and let the reader know the data is loading.
    dataframe_load_state = st.text('Creating dataframe...done!')
    # Load 10,000 rows of data into the dataframe.
    Boston=load_dataframe()
    # Notify the reader that the data was successfully loaded.
    dataframe_load_state.text('Loading dataframe...done!')
    # print the top 5 of the created Dataframe
    st.write('Below is a snapshot is listed of the first 5 records from the created dataframe')
    st.write(Boston.head(5))
    # creating test and training set
    x = Boston.loc[:, Boston.columns != 'MEDV'].values
    y = Boston.loc[:, Boston.columns == 'MEDV'].values
    x_train, x_test, y_train, y_test = train_test_split (Boston[boston.feature_names],y, test_size = 0.25, random_state=34)
    st.title('Setting the benchmark: Fitting a simple linear regression model')
    st.write('Note that this model is previously fitted and loaded here, due to performance reasons')
    lin_reg = pickle.load(open("lin_reg.dat", "rb")).fit(x_train, y_train)
    r_sq = lin_reg.score(x_train,y_train)
    r_sq_t = lin_reg.score(x_test,y_test)
    y_pred = lin_reg.predict(x_test)
    st.write('The benchmark model on this dataset yields the following result:')
    st.write('R-squared: Training Set',round(r_sq,2))
    st.write('R-squared: Test Set',round(r_sq_t,2))
    st.write(f'Since the benchmark model has an R-square of {round(r_sq_t,2)} on the test set, we will continue with a linear kernel')
    st.write('For easier comparison with other models, the Root Mean Squared Error (RMSE) score is also reported')
    st.write('RMSE of baseline model on test set:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
    # Building the dashboard on XGBOOST model:
    st.title('Model the Boston Housing Dataset using XGBOOST')
    # creating DMatrices for XGBOOST application
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=boston.feature_names)
    dtest  = xgb.DMatrix(x_test, label=y_test, feature_names=boston.feature_names)
    # Loading the cross-validated tuned XGBOOST model
    loaded_model = pickle.load(open("xgboost_cv_best_pickle.dat", "rb"))
    st.write('Note that this model is previously fitted and loaded here, due to performance reasons')
    loaded_predictions = loaded_model.predict(dtest)
    st.write('RMSE of the XGBoost model on test set:', round(np.sqrt(metrics.mean_squared_error(y_test, loaded_predictions)),2))
    st.write(f'This means that the model, on average, has an error in predicting the median house value of: {round(np.sqrt(metrics.mean_squared_error(y_test, loaded_predictions)),2)}  Times $1.000.')
    st.write(f'This model scores  {round(abs(((round(np.sqrt(metrics.mean_squared_error(y_test, loaded_predictions)),2)- round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2) )/ round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))*100),2)} percent better on the unseen test data than the benchmark model.')
    st.title('Explaining the XGBoost model.. to a wider audience')
    st.write('below, all seperate decision trees that have been build by training the model can be reviewed')
    ntree=st.number_input('Select the desired record for detailed explanation on the training set'
                                       , min_value=min(range(loaded_model.best_iteration))
                                       , max_value=max(range(loaded_model.best_iteration+1))
                                       )
    tree=xgb.to_graphviz(loaded_model,num_trees=ntree)
    st.graphviz_chart(tree)
    st.write('Using the standard XGBOOST importance plot feature, exposes the fact that the most important feature is not stable, select'
             ' different importance types using the selectbox below')
    importance_type = st.selectbox('Select the desired importance type', ('weight','gain','cover'),index=0)
    importance_plot = xgb.plot_importance(loaded_model,importance_type=importance_type)
    pl.title ('xgboost.plot_importance(best XGBoost model) importance type = '+ str(importance_type))
    st.pyplot(bbox_inches='tight')
    pl.clf()
    st.write('To handle this inconsitency, SHAP values give robust details, among which is feature importance')
    explainer = shap.TreeExplainer(loaded_model)
    shap_values = explainer.shap_values(x_train)
    pl.title('Assessing feature importance based on Shap values')
    shap.summary_plot(shap_values,x_train,plot_type="bar",show=False)
    st.pyplot(bbox_inches='tight')
    pl.clf()
    st.write('SHAP values can also be used to represent the distribution of the training set of the respectable'
             'SHAP value in relation with the Target value, in this case the Median House Value (MEDV)')
    pl.title('Total distribution of observations based on Shap values, colored by Target value')
    shap.summary_plot(shap_values,x_train,show=False)
    st.pyplot(bbox_inches='tight')
    pl.clf()
    st.write('Another example of SHAP values is for GDPR regulation, one should be able to give detailed information as to'
              ' why a specific prediction was made.')
    expectation = explainer.expected_value
    individual = st.number_input('Select the desired record from the training set for detailed explanation.'
                                           , min_value=min(range(len(x_train)))
                                           , max_value=max(range(len(x_train))))
    predicted_values = loaded_model.predict(dtrain)
    real_value = y_train[individual]
    st.write('The real median house value for this individual record is: '+str(real_value))
    st.write('The predicted median house value for this individual record is: '+str(predicted_values[individual]))
    st.write('This prediction is calculated as follows: '
              'The average median house value: ('+str(expectation)+')'+
               ' + the sum of the SHAP values. ')
    st.write(  'For this individual record the sum of the SHAP values is: '+str(sum(shap_values[individual,:])))
    st.write(  'This yields to a predicted value of median house value of:'+str(expectation)+' + '+str(sum(shap_values[individual,:]))+
               '= '+str(expectation+(sum(shap_values[individual,:]))))
    st.write('Which features caused this specific prediction? features in red increased the prediction, in blue decreased them')
    shap.force_plot(explainer.expected_value, shap_values[individual,:],x_train.iloc[individual,:],matplotlib=True,show=False
                    ,figsize=(16,5))
    st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
    pl.clf()
    st.write('In the plot above, the feature values are shown. The SHAP values are represented by the length of the specific bar.'
             'However, it is not quite clear what each single SHAP value is exactly, this can be seen below, if wanted.')
    if st.button('Click here to see a drilldown of the SHAP values'):
        shap_table=pd.DataFrame(shap_values,columns=loaded_model.feature_names)
        st.table(shap_table.iloc[individual])
    st.title('Developing a deeper understanding of the data using SHAP: Interaction effects')
    st.write('When selecting features below, note that the alglorithm automatically plots the slected feature, with the feature that'
             ' it most likely interacts with. However, the final judgement lies in the eyes of the beholder. Typically, when there is'
             ' an interaction effect, points diverge strongly')

    st.write('In the slider below, select the number of features to inspect for possible interaction effects.'
             'These are ordered based on feature importance in the model.')
    ranges = st.slider('Please select the number of features',min_value=min(range(len(x_train.columns)))+1, max_value=max(range(len(x_train.columns)))+1,value=1)
    if ranges-1 == 0:
         st.write('you have selected the most importance feature')
    elif ranges == len(x_train.columns):
            st.write('you have selected all possible features')
    else:
         st.write('you have selected the top:',ranges,'important features')
    for rank in range(ranges):
         ingest=('rank('+str(rank)+')')
         shap.dependence_plot(ingest,shap_values,x_train,show=False)
         st.pyplot()
         pl.clf()
    st.write('Conclusion: It is to my best judgement that there are no significant interaction effects within the features of this model.')
    st.title(' Understanding groups: t-SNE cluster analysis' )
    st.write('Below is an interactive T-SNE cluster plot. If you drag your mouse whilst holding the left mouse button, characteristics of'
             ' all the features are automatically shown. Herewith one can get meaningfull insights of different groups in for example:'
             ' targeting and communication in a marketing context.')
    st.write('Note that the target variable here is the sum of all the SHAP-values for that given datapoint. Furthermore, it is binned into 4 equal groups,for better interpretability. each seperate color stands for a specific group,'
             ' where red signals the highest 25% of (predicted) median house values in Boston.')
    shap_embedded = TSNE(n_components=2, perplexity=25,random_state=34).fit_transform(shap_values)
    source=x_train.copy()
    source.insert(len(source.columns), "TSNE-1", shap_embedded[:,0])
    source.insert(len(source.columns), "TSNE-2", shap_embedded[:,1])
    source.insert(len(source.columns), "TARGET", shap_values.sum(1).astype(np.float64))
    bins = [0,25, 50, 75, 100]
    labels = ['lowest 25%','25 to 50%','50-75%','highest 25%']
    source['TARGET_BINNED'] = pd.cut(source['TARGET'], bins=4,labels=labels).astype(str)
    brush = alt.selection(type='interval',resolve='global')
    points_TSNE = alt.Chart(source).mark_point().encode(
           x='TSNE-1:Q',
           y='TSNE-2:Q',
           color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
    ).add_selection(
           brush
    ).properties(
        width=500,
        height=250
    )
    points_LSTAT = alt.Chart(source).mark_circle().encode(
         x='LSTAT:Q',
         y='TARGET:Q',
         color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray')),
         # size='TARGET'
     ).add_selection(
          brush
    ).properties(
        width=500,
        height=250
    )
    points_RM = alt.Chart(source).mark_circle().encode(
         x='RM:Q',
         y='TARGET:Q',
         color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
     ).add_selection(
          brush
     ).properties(
        width=500,
        height=250
    )
    points_DIS = alt.Chart(source).mark_point().encode(
         x='DIS:Q',
         y='TARGET:Q',
         color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
     ).add_selection(
         brush
     ).properties(
        width=500,
        height=250
    )
    points_PTRATIO = alt.Chart(source).mark_point().encode(
        x='PTRATIO:Q',
        y='TARGET:Q',
        color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray')),
    ).add_selection(
        brush
    ).properties(
        width=500,
        height=250
    )
    points_AGE = alt.Chart(source).mark_point().encode(
        x='AGE:Q',
        y='TARGET:Q',
        color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
    ).add_selection(
        brush
    ).properties(
        width=500,
        height=250
    )
    points_CRIM = alt.Chart(source).mark_point().encode(
        x='CRIM:Q',
        y='TARGET:Q',
        color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
    ).add_selection(
        brush
    ).properties(
        width=500,
        height=250
    )
    points_B = alt.Chart(source).mark_point().encode(
        x='B:Q',
        y='TARGET:Q',
        color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
    ).add_selection(
        brush
    ).properties(
        width=500,
        height=250
    )
    points_NOX = alt.Chart(source).mark_point().encode(
        x='NOX:Q',
        y='TARGET:Q',
        color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
    ).add_selection(
        brush
    ).properties(
        width=500,
        height=250
    )
    points_TAX = alt.Chart(source).mark_point().encode(
        x='TAX:Q',
        y='TARGET:Q',
        color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
    ).add_selection(
        brush
    ).properties(
        width=500,
        height=250
    )
    points_RAD = alt.Chart(source).mark_point().encode(
        x='RAD:Q',
        y='TARGET:Q',
        color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
    ).add_selection(
        brush
    ).properties(
        width=500,
        height=250
    )
    points_INDUS = alt.Chart(source).mark_point().encode(
        x='INDUS:Q',
        y='TARGET:Q',
        color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
    ).add_selection(
        brush
    ).properties(
        width=500,
        height=250
    )
    points_CHAS = alt.Chart(source).mark_point().encode(
        x='CHAS:Q',
        y='TARGET:Q',
        color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
    ).add_selection(
        brush
    ).properties(
        width=500,
        height=250
    )
    points_ZN = alt.Chart(source).mark_point().encode(
        x='ZN:Q',
        y='TARGET:Q',
        color=alt.condition(brush, 'TARGET_BINNED:N', alt.value('lightgray'))
    ).add_selection(
        brush
    ).properties(
        width=500,
        height=250
    )
    st.altair_chart(points_TSNE & points_LSTAT & points_RM & points_DIS & points_PTRATIO & points_AGE & points_CRIM
                    & points_B & points_NOX & points_TAX & points_RAD & points_INDUS & points_CHAS & points_ZN )
if __name__== '__main__':
    main()


