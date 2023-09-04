from flask import render_template, request, session, redirect
from flask import current_app as app
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import dtreeviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
from sklearn.linear_model import LogisticRegression
import numpy as np
import tensorflow as tf
import joblib

data = pd.read_csv("C:\\Users\\aarus\\Desktop\\ai proj\\static\\parkinsons.data", sep=',', index_col='name') 
X = data.drop('status', axis=1)
X.head()
y=data['status']
y.head()


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = 3*cm.max()/4
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Embed the plot in the HTML template
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url
@app.route('/')
def root():
    return render_template("home.html")



@app.route('/plot')
def plot():
    # Create your plot using matplotlib
    

    plt.figure(figsize=(10, 10))
    heatmap = sns.heatmap(data.corr()[['status']].sort_values(by='status', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Features Correlating with Parkinson existence', fontdict={'fontsize':18}, pad=16)
    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Embed the plot in the HTML template
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('plot.html', plot_url=plot_url)

@app.route('/dataset')
def dataset():
    return render_template("dataset.html")


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/decision-tree')
def dt():
    return render_template("dt.html")

@app.route('/logistic-regression')
def lor():
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    coef=abs(lr.coef_[0])
    best_columns=X.columns[np.argsort(-1*coef)[0:5]]
    str=""
    for i in best_columns:
        str+=i
        str+=','
    return render_template("lr.html", accuracy_score=accuracy_score(y_test, y_pred),bp=str)

@app.route("/neural-network")
def nn():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(22,)),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    accuracy = accuracy_score(y_test, y_pred)
    plot_url=plot_confusion_matrix(confusion_matrix(y_test, y_pred),classes=[ "Not Parkinson", " Parkinson"],title='Confusion matrix')
   
    return render_template("nn.html",acc=accuracy,url=plot_url)

@app.route('/predict', methods=['GET','POST'])
def pred():

    model = joblib.load("trained_model.pkl")
    if request.method=="GET":
        return render_template("predict.html")

    if request.method=="POST":
        if "person" in request.form:
            if request.form['person']=='person1':
                X_new = np.array([[148.14300,155.98200,135.04100,0.00392,0.00003,0.00204,0.00231,0.00612,0.01450,0.13100,0.00725,0.00876,0.01263,0.02175,0.00540,23.68300,0.398499,0.778349,-5.711205,0.240875,2.845109,0.192730]])
            if request.form['person']=='person2':
                X_new = np.array([[110.70700,122.61100,105.00700,0.00516,0.00005,0.00277,0.00289,0.00831,0.02215,0.20600,0.01284,0.01219,0.01715,0.03851,0.00472,25.19700,0.463514,0.807217,-5.477592,0.315074,1.862092,0.228624]])
            if request.form['person']=='person3':
                X_new = np.array([[197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569]])
            if request.form['person']=='person4':
                X_new = np.array([[178.28500,442.82400,82.06300,0.00462,0.00003,0.00157,0.00194,0.00472,0.01279,0.12900,0.00617,0.00744,0.01151,0.01851,0.00856,25.02000,0.470422,0.655239,-4.913137,0.393056,2.816781,0.251972]])
            if request.form['person']=='person5':
                X_new = np.array([[117.22600,123.92500,106.65600,0.00417,0.00004,0.00186,0.00270,0.00558,0.01909,0.17100,0.00864,0.01223,0.01949,0.02592,0.00955,23.07900,0.603515,0.669565,-5.619070,0.191576,2.027228,0.215724]])

        y_pred=model.predict(X_new)
        y_pred = (y_pred > 0.5).astype(int)
        if y_pred==1:
            p="Positive for Parkinson's"
        if y_pred==0:
            p="Negative for Parkinson's"
        return render_template("predict.html",p=p)