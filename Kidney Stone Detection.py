import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
import pickle
import seaborn as sns

from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
Categories=['Normal','Stone']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir=r"DATASET"

#create file paths by combining the datadir (data directory) with the filenames 'flat_data.npy
flat_data_file = os.path.join(datadir, 'flat_data.npy')
target_file = os.path.join(datadir, 'target.npy')

if os.path.exists(flat_data_file) and os.path.exists(target_file):
    # Load the existing arrays
    flat_data = np.load(flat_data_file)
    target = np.load(target_file)
else:
    #path which contains all the categories of images
    for i in Categories:
        print(f'loading... category : {i}')
        path=os.path.join(datadir,i)
        #create file paths by combining the datadir (data directory) with the i
        for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))#Reads the image using imread.
            img_resized=resize(img_array,(150,150,3)) #Resizes the image to a common size of (150, 150, 3) pixels.
            flat_data_arr.append(img_resized.flatten()) #Flattens the resized image array and adds it to the flat_data_arr.
            target_arr.append(Categories.index(i)) #Adds the index of the category to the target_arr.
            #this index is being used to associate the numerical representation of the category (index) with the actual image data. This is often done to provide labels for machine learning algorithms where classes are represented numerically. In this case, 'ORGANIC' might correspond to label 0, and 'NONORGANIC' might correspond to label 1.
            print(f'loaded category:{i} successfully')
            #After processing all images, it converts the lists to NumPy arrays (flat_data and target).
            flat_data=np.array(flat_data_arr)
            target=np.array(target_arr)
    # Save the arrays(flat_data ,target ) into the files(flat_data.npy,target.npy)
    np.save(os.path.join(datadir, 'flat_data.npy'), flat_data)
    np.save(os.path.join(datadir, 'target.npy'), target)
#dataframe
df=pd.DataFrame(flat_data)
df['Target']=target #associated the numerical representation of the category (index) with the actual image data
df
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=df, x='Target')
plt.xlabel('Target', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Count Plot for Target', fontsize=14)

# Add count labels on top of the bars
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.show()
#input data
x=df.iloc[:,:-1]
x
#output data
y=df.iloc[:,-1]
y

# Splitting the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.40,random_state=77)

#DECISIONTREE                                                                                                                                       from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
filename = 'DT_Classifier.pkl'
if os.path.exists('DT_Classifier.pkl'):
    # Load the trained model from the Pickle file
    with open(filename, 'rb') as DT_Model_pkl:
        DT = pickle.load(DT_Model_pkl)
        y_pred=DT.predict(x_test)
        Acc=accuracy_score(y_test,y_pred)*100
        print("Accuracy",Acc)
else:
    DT.fit(x_train,y_train)
    y_pred=DT.predict(x_test)
    Acc=accuracy_score(y_test,y_pred)*100
    print("Accuracy",Acc)
    # Dump the trained rf classifier with Pickle
    filename = 'DT_Classifier.pkl'
    # Open the file to save as pkl file
    DT_Model_pkl = open(filename, 'wb')
    #when you use 'wb' as the mode when opening a file, you are telling Python to open the file in write mode and treat it as a binary file. This is commonly used when saving non-textual data, such as images, audio, or serialized objects like machine learning models
    pickle.dump(DT, DT_Model_pkl)
    #function to serialize and save the rf object (which is your trained Random Forest model) into the Pickle file opened as RF_Model_pkl.
    # Close the pickle instances
    DT_Model_pkl.close()
cm=confusion_matrix(y_test,y_pred)
cm                                                                                                                          class_labels=['Stone','Normal']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("DecisionTreeClassifierConfusion Matrix")
plt.show()
report=classification_report(y_test,y_pred,target_names=class_labels)
print('DecisionTreeClassifier\n',report)
#RANDOMFOREST                                                                                                                                 from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
filename = 'RF_Classifier.pkl'
if os.path.exists('RF_Classifier.pkl'):
    # Load the trained model from the Pickle file
    with open(filename, 'rb') as RF_Model_pkl:
        rf = pickle.load(RF_Model_pkl)
        y_pred=rf.predict(x_test)
        Acc=accuracy_score(y_test,y_pred)*100
        print("Accuracy",Acc)
else:
    rf.fit(x_train,y_train)
    y_pred=rf.predict(x_test)
    Acc=accuracy_score(y_test,y_pred)*100
    print("Accuracy",Acc)
    # Dump the trained rf classifier with Pickle
    filename = 'RF_Classifier.pkl'
    # Open the file to save as pkl file
    RF_Model_pkl = open(filename, 'wb')
    #when you use 'wb' as the mode when opening a file, you are telling Python to open the file in write mode and treat it as a binary file. This is commonly used when saving non-textual data, such as images, audio, or serialized objects like machine learning models
    pickle.dump(rf, RF_Model_pkl)
    #function to serialize and save the rf object (which is your trained Random Forest model) into the Pickle file opened as RF_Model_pkl.
    # Close the pickle instances
    RF_Model_pkl.close()                                                                                                              Acc=accuracy_score(y_test,y_pred)*100
print("Accuracy",Acc)                                                                                 cm=confusion_matrix(y_test,y_pred)
cm                                                                                                              class_labels=['Stone','Normal']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("RandomForestClassifier Confusion Matrix")
plt.show()                                                                      report=classification_report(y_test,y_pred,target_names=class_labels)
print('RandomForestClassifier\n',report)

path = r"testing"
Categories = {1:'Stone',0:'Normal'}  # Define your categories with corresponding labels
for filename in os.listdir(path):
    img_path = os.path.join(path, filename)  # Construct the complete image path
    img = imread(img_path)
    
    plt.imshow(img)
    plt.show()
    
    img_resize = resize(img, (150, 150, 3))
    l = [img_resize.flatten()]
    
    # Make predictions using your pre-trained model
    prediction = rf.predict(l)[0]
    predicted_category = Categories[prediction]
    print("The predicted image is:", predicted_category)
