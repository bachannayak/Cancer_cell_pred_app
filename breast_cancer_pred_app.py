import random
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
cancer_data=datasets.load_breast_cancer()
feature = cancer_data['data']
label = cancer_data['target']
df_frt = pd.DataFrame(feature , columns = cancer_data['feature_names'])
df_lbl = pd.DataFrame(label , columns = ['label'])
df = pd.concat([df_frt, df_lbl], axis=1)
df = df.sample(frac = 1)
feature = df_frt
label = df_lbl

rfe_feature_df=feature[['mean smoothness', 'mean compactness', 'mean concave points',
       'mean symmetry', 'mean fractal dimension', 'radius error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'worst smoothness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']]

x_train, x_test, y_train, y_test = train_test_split(rfe_feature_df, label, test_size=0.3, random_state=42)

clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(x_train, y_train)
y_pred = clf_gini.predict(x_test)
score=metrics.accuracy_score(y_test, y_pred)*100





@st.cache()
def prediction(mean_smoothness, mean_compactness, mean_concave_points,mean_symmetry, mean_fractal_dimension, radius_error,smoothness_error, compactness_error, concavity_error,concave_points_error, worst_smoothness, worst_concavity,worst_concave_points, worst_symmetry, worst_fractal_dimension):
  species = clf_gini.predict([[mean_smoothness, mean_compactness, mean_concave_points,mean_symmetry, mean_fractal_dimension, radius_error,smoothness_error, compactness_error, concavity_error,concave_points_error, worst_smoothness, worst_concavity,worst_concave_points, worst_symmetry, worst_fractal_dimension]])
  species = species[0]
  if (species > 0.5):
    return "Benign"
  else:
    return "Malignant"

# S10.3: Perform this activity in Sublime editor after adding the above code. 
# Add title widget
st.title("Breast Cancer Prediction App")  

# Add 15 sliders and store the value returned by them in 15 separate variables.

mean_smoothness=st.slider("mean_smoothness", 0.053 , 0.163)
mean_compactness=st.slider("mean_compactness", 0.019 , 0.345)
mean_concave_points=st.slider("mean_concave_points", 0.0 ,0.201)
mean_symmetry=st.slider("mean_symmetry", 0.106 ,0.304)
mean_fractal_dimension=st.slider("mean_fractal_dimension", 0.05 ,0.097)
radius_error=st.slider("radius_error", 0.112 ,2.873)
smoothness_error=st.slider("smoothness_error", 0.002,0.031)
compactness_error=st.slider("compactness_error", 0.002,0.135)
concavity_error=st.slider("concavity_erro", 0.0,0.396)
concave_points_error=st.slider("concave_points_erro", 0.0,0.053)
worst_smoothness=st.slider("worst_smoothness", 0.071, 0.223)
worst_concavity=st.slider("worst_concavity", 0.0 ,1.252)
worst_concave_points=st.slider("worst_concave_points", 0.0 ,0.291)
worst_symmetry=st.slider("worst_symmetry", 0.156 ,0.664)
worst_fractal_dimension=st.slider("worst_fractal_dimension", 0.055,0.208)


# When 'Predict' button is pushed, the 'prediction()' function must be called 
# and the value returned by it must be stored in a variable, say 'species_type'. 
# Print the value of 'species_type' and 'score' variable using the 'st.write()' function.
if st.button("Predict"):
	species_type = prediction(mean_smoothness, mean_compactness, mean_concave_points,mean_symmetry, mean_fractal_dimension, radius_error,smoothness_error, compactness_error, concavity_error,concave_points_error, worst_smoothness, worst_concavity,worst_concave_points, worst_symmetry, worst_fractal_dimension)
	st.write("Predicted Tumor is ", species_type,'type')
	st.write("Accuracy score of this model is:", score)