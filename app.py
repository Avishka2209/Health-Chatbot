from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
app = Flask(__name__, static_url_path='', static_folder='.')

# Load datasets
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')

# Extract features and target variable
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Encoding target variable
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# Define symptom dictionary
symptoms_dict = {symptom.lower(): index for index, symptom in enumerate(x)}

# Load symptom descriptions
description_df = pd.read_csv('symptom_Description.csv', header=None)
description_list = dict(zip(description_df[0].str.lower(), description_df[1]))

# Load precaution details
precaution_df = pd.read_csv('symptom_precaution.csv', header=None)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/diagnose", methods=["POST"])
def diagnose():
    symptoms = request.json["symptoms"]
    input_vector = np.zeros(len(cols))
    for symptom in symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
        else:
            return jsonify({"error": f"Symptom '{symptom}' not found in the database."}), 400

    pred_diseases = clf.predict([input_vector])
    disease_name = le.inverse_transform([pred_diseases[0]])[0]
    description = description_list.get(disease_name.lower(), "Description not available")
    #precautions = precaution_df[precaution_df[0].str.lower() == disease_name.lower()]
    #precautions = precautions[1].tolist() if not precautions.empty else ["Precautions not available"]
    precautions = precaution_df.loc[precaution_df[0].str.lower() == disease_name.lower(), [1, 2, 3,4]]
    precautions = precautions.values.tolist() if not precautions.empty else [["Precautions not available"]]



    # Find other possibilities based on provided symptoms
    other_possibilities = set()
    for index, row in training.iterrows():
        if all(row[symptom] == 1 for symptom in symptoms):
            other_possibilities.add(row['prognosis'])
    other_possibilities.discard(disease_name)  # Remove the predicted disease

    return jsonify({
        "disease_name": disease_name,
        "description": description,
        "precautions": precautions,
        "other_possibilities": list(other_possibilities)
    })

if __name__ == "__main__":
    app.run(debug=True)
