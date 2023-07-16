import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor,HistGradientBoostingRegressor
#import xgboost
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, r2_score
import streamlit as st
import time
#import shap
#import matplotlib as mt
from jinja2 import Template
import seaborn as sns


def train(data=None,problem="Regression",model="LinearRegression",label=None):

    df = pd.read_csv(data)

    target = df[label].copy()
    features = df.drop(label, axis=1)

    X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.20,random_state=42,shuffle=True)

    num_features = []
    cat_features = []
    cols = list(features.columns)
    for i in cols:
        if df[i].dtypes == "object":
            cat_features.append(i)
        else:
            num_features.append(i)

    if problem == "Regression":
        trf = ColumnTransformer([("num_trf",StandardScaler(),num_features),
                                 ("cat_trf",OneHotEncoder(sparse_output=False),cat_features)])
        
            
        
        
        if model == "LinearRegression":
            final_pipe = Pipeline([("transformers",trf),("reg_model",LinearRegression())])
        elif model == "RandomForestRegressor":
            final_pipe = Pipeline([("transformers",trf),("rf_reg_model",RandomForestRegressor(random_state=42))])
        else:
            final_pipe = Pipeline([("transformers",trf),("reg_model",HistGradientBoostingRegressor(random_state=42))])

        final_pipe.fit(X_train,y_train)

        final_pipe.fit(X_train,y_train)

        #model = pickle.dump(final_pipe,open("regression_model","wb"))

        #y_hat = model.predict(X_train)
        model_name = str(model)+".pkl"
        model = pickle.dump(final_pipe,open(model_name,"wb"))


        return final_pipe, X_train,X_test,y_train,y_test, model_name
    if problem == "Classification":
        if model == "GradientBoosting":

            trf = ColumnTransformer([("num_trf",StandardScaler(),num_features),
                                    ("cat_trf",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat_features)])
            
            
            lbl_encd = LabelEncoder()

            lbl_encd.fit(y_train)
            y_train_trf = lbl_encd.transform(y_train)

            y_test_trf = lbl_encd.fit(y_test)
            
            final_pipe = Pipeline([("transformers",trf),("clf_model",GradientBoostingClassifier(random_state=42))])

            final_pipe.fit(X_train,y_train_trf)
            #file = open("model")
            #model = pickle.dump(final_pipe,("","wb"))
            model_name = str(model)+".pkl"
            model = pickle.dump(final_pipe,open(model_name,"wb"))


            return final_pipe, X_train,X_test,y_train_trf,y_test_trf, model_name
        elif model == "LogisticRegression":
            trf = ColumnTransformer([("num_trf",StandardScaler(),num_features),
                                    ("cat_trf",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat_features)])
            
            
            lbl_encd = LabelEncoder()

            lbl_encd.fit(y_train)
            y_train_trf = lbl_encd.transform(y_train)

            y_test_trf = lbl_encd.fit(y_test)
            
            final_pipe = Pipeline([("transformers",trf),("clf_model",LogisticRegression(random_state=42))])

            final_pipe.fit(X_train,y_train_trf)
            #file = open("model")
            model_name = str(model)+".pkl"
            pickle.dump(final_pipe,open(model_name,"wb"))

            return final_pipe, X_train,X_test,y_train_trf,y_test_trf,model_name


def predict(model=None,x=None):

    #m = pickle.load(open(model,"rb"))
    y_hat = model.predict(x)

    return y_hat

def evaluate(y_true,y_pred, problem="Regression"):

    if problem == "Regression":
        metric = r2_score(y_true,y_pred)
        return metric
    else:
        metric = classification_report(y_true,y_pred,output_dict=True)
        met_df = pd.DataFrame(metric).transpose()
        file = met_df.to_csv().encode('utf-8')

        return file
    


st.title("No Code Machine Learning Studio :six_pointed_star:")
st.caption("An application developed by Indranil Bhattacharyya")

st.image(image="https://editor.analyticsvidhya.com/uploads/76748pycaret2-features.png")
st.caption("Plug & Play Portal for Machine Learing")


#sidebar: initiating
with st.sidebar:
    st.header("Model Configurations will appear here")



prob_type = st.selectbox(label="Please select your ML problem type: ",options=("Regression","Classification"))

train_data = st.file_uploader(label="Please upload your training dataset",type=["csv"])


if prob_type == "Classification":

    model = st.selectbox(label="Plase Select your classification model: ", options=("GradientBoosting","LogisticRegression"))
else:
    model = st.selectbox(label="Plase Select your classification model: ", options=("LinearRegression","RandomForestRegressor","HistGradientBoostingRegressor"))
with st.sidebar:
    st.subheader("Selected Problem Type(Regression/Classification): ")
    st.write(prob_type)
    if prob_type == "Regression":
        st.caption("In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships between a dependent variable and one or more independent variables.")
    else:
        st.caption('The Classification algorithm is a Supervised Learning technique that is used to identify the category of new observations on the basis of training data. In Classification, a program learns from the given dataset or observations and then classifies new observation into a number of classes or groups. Such as, Yes or No, 0 or 1, Spam or Not Spam, cat or dog, etc. Classes can be called as targets/labels or categories.')
    st.code(model,language='python')

    #def explain(model="LinearRegression",train_data=None,test_data=None):
    #explainer = shap.LinearExplainer(model,train_data,feature_dependence=False)
    #    shap_values = explainer.shap_values(test_data)

    #    shap.summary_plot(shap_values,test_data,plot_type="violin",show=False)
    #    mt.pyplot.gcf().axes[-1].set_box_aspect(10)


y = st.text_input("Please write your target column name:  ")
#num_f = st.text_input("Please write your numerical feature names(separted by ","): ").split(",")
#cat_f = st.text_input("Please write your categorical feature names(separted by ","): ").split(",")

if st.button("Train"):
    
    time.sleep(1)
    
    #for classification
    if prob_type=="Classification":
        with st.progress(10,"Discovering the dataset..."):
            time.sleep(0.5)
            st.progress(20, "Applying the preprocessing steps...")
            time.sleep(1)
            st.progress(25,"Training engine has started...")
            st.progress(50, "Training the model...")
            model_, X_train,X_test,y_train,y_test,model_name = train(data=train_data,problem=prob_type,model=model, label=y)
            time.sleep(2)
            st.progress(75, "Training complete...")
            st.progress(85, "Evaluating model performance...")
            st.progress(90, "Generating Classification report...")
            time.sleep(1)
            st.progress(100, "Complete! :100:")

        #for printing model defintion 
        with st.sidebar:
            st.divider()
            st.markdown("Model Parameters: ")
            st.divider()
            
            st.code(model_.get_params(deep=False),language="python")
            st.divider()
            
            st.code(model_.decision_function(X_train),language="python")
            st.divider()

            st.code(model_[:-1].get_feature_names_out())

        y_hat_train = predict(model_,X_train)
        y_hat_test = predict(model_,X_test)
        report = evaluate(y_train,y_hat_train,prob_type)

        st.download_button(label="Click here to download the report",data=report, mime="text/csv")
        time.sleep(2)
        st.write("Classification report of testing dataset: ")
        report_test = evaluate(y_train,y_hat_train,prob_type)
        st.download_button(key="test",label="Click here to download the report",data=report_test, mime="text/csv")
        st.success("Report generated successfully! :beers:")
        time.sleep(5)
        model_download = st.radio("Do you want to download the model?",options=["Yes","No, who wants the model!"])
        
        if model_download == "Yes":
            if st.download_button("Click Here to Download!",data=model_name,file_name=str(model_name),mime="binary_contents"):
                time.sleep(1)
        time.sleep(20)
    
    #for regression 
    else:
        with st.progress(10,"Discovering the dataset..."):
            time.sleep(0.5)
            st.progress(20, "Applying the preprocessing steps...")
            time.sleep(1)
            st.progress(25,"Training engine has started...")
            st.progress(50, "Training the model...")
            model_, X_train,X_test,y_train,y_test, model_name = train(data=train_data,problem=prob_type,model=model, label=y)
            time.sleep(2)
            st.progress(75, "Training complete...")
            st.progress(85, "Evaluating model performance...")
            st.progress(90, "Generating Regression metrics...")
            time.sleep(1)
            st.progress(100, "Complete! :100:")
        
        #for printing model defintion 
        with st.sidebar:
            st.divider()
            st.write("Model Parameters: ")
            st.code(model_.get_params(deep=False),language="python")
            
        y_hat_train = predict(model_,X_train)
        y_hat_test = predict(model_,X_test)
        #st.write("r2 score on training set: ")
        st.metric(label="r2 score on training set: ",value=evaluate(y_train,y_hat_train))
        #st.write("r2 score on test set: ")
        time.sleep(0.5)

        st.metric(label="r2 score on test set: ",value=evaluate(y_test,y_hat_test,prob_type)) 
        st.success("Metrics generated successfully! :beers:")
        model_download = st.radio("Do you want to download the model?",options=["Yes","No, who wants the model!"])
        
        if model_download == "Yes":
            if st.download_button("Click Here to Download!",data=model_name,file_name=str(model_name),mime="binary_contents"):
                time.sleep(1)

        reg = '''
        No Code Machine Learning Platform
        A Plug & Play Platform for MachineLearning
                -made by Indranil Bhattacharyya

        UserName: {{ User_name }}                                      Date: {{ date }}

        Model Report: 

        Problem Type: {{ problem_type }}

        In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships between a dependent variable and one or more independent variables. 
        

        Algorithm Used: {{ algo }}

        Target Column: {{ target }}

        R2 Score: {{ r2_score }}
        R2 is a measure of the goodness of fit of a model. In regression, the R2 coefficient of determination is a statistical measure of how well the regression predictions approximate the real data points. An R2 of 1 indicates that the regression predictions perfectly fit the data.

            Values of R2 outside the range 0 to 1 occur when the model fits the data worse than the worst possible least-squares predictor (equivalent to a horizontal hyperplane at a height equal to the mean of the observed data). This occurs when a wrong model was chosen, or nonsensical constraints were applied by mistake.
        '''
        
        reg_temp = Template(reg)
        teamplate = reg_temp.render(problem_type=prob_type,algo=model,target=y,r2_score=evaluate(y_train,y_hat_train))
        print(teamplate)
        with open("Regression Report.pdf","w") as f:
            f.write(teamplate)
            f.close()
        
        st.success("Your task is completed. Have a nice Day!",icon="✅")
        time.sleep(1)
        
        
            
        
        
     
        
        






            
        

        