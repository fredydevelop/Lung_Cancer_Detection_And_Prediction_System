#Importing the dependencies
import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import streamlit as st
import base64
import pickle as pk




#configuring the page setup
st.set_page_config(page_title='Lung Cancer detection system',layout='centered')

#selection=option_menu(menu_title="Main Menu",options=["Single Prediction","Multi Prediction"],icons=["cast","book","cast"],menu_icon="house",default_index=0)
with st.sidebar:
    st.title("Home Page")
    selection=st.radio("select your option",options=["Predict for a Single-Patient", "Predict for Multi-Patient"])


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your prediction</a>'
    return href


#single prediction function
def LungDetector(givendata):
    
    loaded_model=pk.load(open("The_Latest_Lung Cancer_Model.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    std_scaler_loaded=pk.load(open("my_saved_std_scaler.pkl", "rb"))
    std_X_resample=std_scaler_loaded.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_X_resample)
    if prediction==1 or prediction=="1":
      return "Lung Cancer is present"
    else:
      return "No Lung Cancer"
    
 
#main function handling the input
def main():
    st.header("Lung Cancer Detection System")
    
    #getting user input
    
    age = st.slider('Patient age', 0, 200, key="ageslide")
    st.write("Patient is", age, 'years old')

    option1 = st.selectbox('Gender',("",'Male' ,'Female'),key="gender")
    if (option1=='Male'):
        Gender=1
    else:
        Gender=0

    option2 = st.selectbox('SMOKING',("",'Yes' ,'No'),key="SMOKING")
    if (option2=='Yes'):
        SMOKING=1
    else:
        SMOKING=0

    option3 = st.selectbox('YELLOW_FINGERS',("",'Yes' ,'No'),key="YELLOW_FINGERS")
    if (option3=='YES'):
        YELLOW_FINGERS=1
    else:
        YELLOW_FINGERS=0

    option4 = st.selectbox('ANXIETY',("",'Yes' ,'No'),key="ANXIETY")
    if (option4=='YES'):
        ANXIETY=1
    else:
        ANXIETY=0


    option5 = st.selectbox('PEER_PRESSURE',("",'Yes' ,'No'),key="PEER_PRESSURE")
    if (option5=='YES'):
        PEER_PRESSURE=1
    else:
        PEER_PRESSURE=0



    
#
    option14 = st.selectbox('CHRONIC DISEASE',("",'Yes' ,'No'),key="CHRONIC_DISEASE")
    if (option14=='YES'):
        CHRONIC_DISEASE=1
    else:
        CHRONIC_DISEASE=0


    

    option6 = st.selectbox('FATIGUE',("",'Yes' ,'No'),key="FATIGUE")
    if (option6=='YES'):
        FATIGUE=1
    else:
        FATIGUE=0


    option7 = st.selectbox('ALLERGY',("",'Yes' ,'No'),key="ALLERGY")
    if (option7=='YES'):
        ALLERGY=1
    else:
        ALLERGY=0


    option8 = st.selectbox('WHEEZING',("",'Yes' ,'No'),key="WHEEZING")
    if (option8=='YES'):
        WHEEZING=1
    else:
        WHEEZING=0

    

    option9 = st.selectbox('ALCOHOL_CONSUMING',("",'Yes' ,'No'),key="ALCOHOL_CONSUMING")
    if (option9=='YES'):
        ALCOHOL_CONSUMING=1
    else:
        ALCOHOL_CONSUMING=0

    


    option10 = st.selectbox('COUGHING',("",'Yes' ,'No'),key="COUGHING")
    if (option10=='YES'):
        COUGHING=1
    else:
        COUGHING=0


    

    option11 = st.selectbox('SHORTNESS OF BREATH',("",'Yes' ,'No'),key="SHORTNESS_OF_BREATH")
    if (option11=='YES'):
        SHORTNESS_OF_BREATH=1
    else:
        SHORTNESS_OF_BREATH=0





    option12 = st.selectbox('SWALLOWING DIFFICULTY',("",'Yes' ,'No'),key="SWALLOWING_DIFFICULTY")
    if (option12=='YES'):
        SWALLOWING_DIFFICULTY=1
    else:
        SWALLOWING_DIFFICULTY=0


    

    option13 = st.selectbox('CHEST_PAIN',("",'Yes' ,'No'),key="CHEST_PAIN")
    if (option13=='YES'):
        CHEST_PAIN=1
    else:
        CHEST_PAIN=0
    



    st.write("\n")
    st.write("\n")





    detectionResult = ''#for displaying result
    
    # creating a button for Prediction
    if age!="" and option1!=""  and option2!=""  and option3!=""  and option4!="" and option5!="" and option6!="" and option7 !=""and  option8 !="" and option9!="" and option10 !="" and option11 !="" and option12 !="" and  option13 !="" and option14 !="" and st.button('Predict'):
        detectionResult = LungDetector([age,Gender,SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE , ALLERGY , WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN])
        st.success(detectionResult)


def multi(input_data):
    loaded_model=pk.load(open("The_Latest_Lung Cancer_Model.sav", "rb"))
    dfinput = pd.read_csv(input_data,header=None)
    dfinput=dfinput.iloc[1:].reset_index(drop=True)

    st.header('A view of the uploaded dataset')
    st.markdown('')
    st.dataframe(dfinput)

    dfinput=dfinput.values
    std_scaler_loaded=pk.load(open("my_saved_std_scaler.pkl", "rb"))
    std_dfinput=std_scaler_loaded.transform(dfinput)
    
    
    predict=st.button("predict")


    if predict:
        prediction = loaded_model.predict(std_dfinput)
        interchange=[]
        for i in prediction:
            if i==1:
                newi="Lung Cancer Detected"
                interchange.append(newi)
            elif i==0:
                newi="No Lung Cancer"
                interchange.append(newi)
            
        st.subheader('Here is your prediction')
        prediction_output = pd.Series(interchange, name='Lung Detection results')
        prediction_id = pd.Series(np.arange(len(interchange)),name="Patient_ID")
        dfresult = pd.concat([prediction_id, prediction_output], axis=1)
        st.dataframe(dfresult)
        st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        

if selection =="Predict for a Single-Patient":
    main()

if selection == "Predict for Multi-Patient":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #---------------------------------#
    # Prediction
    #--------------------------------
    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    st.header('Upload your csv file here')
    uploaded_file = st.file_uploader("", type=["csv"])
    #--------------Visualization-------------------#
    # Main panel
    
    # Displays the dataset
    if uploaded_file is not None:
        #load_data = pd.read_table(uploaded_file).
        multi(uploaded_file)
    else:
        st.info('Upload your dataset !!')
    
