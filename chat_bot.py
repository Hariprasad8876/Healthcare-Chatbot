# Install necessary libraries if not already installed
!pip install pyttsx3

import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import tkinter as tk
from tkinter import scrolledtext

# Suppress DeprecationWarnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load datasets
training = pd.read_csv('C:\\Users\\Hari\\Downloads\\healthcare-chatbot-master (1)\\healthcare-chatbot-master\\Data\\Training.csv')
testing = pd.read_csv('C:\\Users\\Hari\\Downloads\\healthcare-chatbot-master (1)\\healthcare-chatbot-master\\Data\\Testing.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

# Mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print(f"Decision Tree Classifier accuracy: {scores.mean()}")

model = SVC()
model.fit(x_train, y_train)
print(f"SVM accuracy: {model.score(x_test, y_test)}")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('voice', "english+f5")
engine.setProperty('rate', 130)

def readn(nstr):
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

# Load additional data files
severityDictionary = {
    'itching': 1,
    'skin_rash': 3,
    'nodal_skin_eruptions': 4,
    'continuous_sneezing': 4,
    'shivering': 5,
    'chills': 3,
    'joint_pain': 3,
    'stomach_pain': 5,
    'acidity': 3,
    'ulcers_on_tongue': 4,
    'muscle_wasting': 3,
    'vomiting': 5,
    'burning_micturition': 6,
    'spotting_urination': 6,
    'fatigue': 4,
    'weight_gain': 3,
    'anxiety': 4,
    'cold_hands_and_feets': 5,
    'mood_swings': 3,
    'weight_loss': 3,
    'restlessness': 5,
    'lethargy': 2,
    'patches_in_throat': 6,
    'irregular_sugar_level': 5,
    'cough': 4,
    'high_fever': 7,
    'sunken_eyes': 3,
    'breathlessness': 4,
    'sweating': 3,
    'dehydration': 4,
    'indigestion': 5,
    'headache': 3,
    'yellowish_skin': 3,
    'dark_urine': 4,
    'nausea': 5,
    'loss_of_appetite': 4,
    'pain_behind_the_eyes': 4,
    'back_pain': 3,
    'constipation': 4,
    'abdominal_pain': 4,
    'diarrhoea': 6,
    'mild_fever': 5,
    'yellow_urine': 4,
    'yellowing_of_eyes': 4,
    'acute_liver_failure': 6,
    'fluid_overload': 6,
    'swelling_of_stomach': 7,
    'swelled_lymph_nodes': 6,
    'malaise': 6,
    'blurred_and_distorted_vision': 5,
    'phlegm': 5,
    'throat_irritation': 4,
    'redness_of_eyes': 5,
    'sinus_pressure': 4,
    'runny_nose': 5,
    'congestion': 5,
    'chest_pain': 7,
    'weakness_in_limbs': 7,
    'fast_heart_rate': 5,
    'pain_during_bowel_movements': 5,
    'pain_in_anal_region': 6,
    'bloody_stool': 5,
    'irritation_in_anus': 6,
    'neck_pain': 5,
    'dizziness': 4,
    'cramps': 4,
    'bruising': 4,
    'obesity': 4,
    'swollen_legs': 5,
    'swollen_blood_vessels': 5,
    'puffy_face_and_eyes': 5,
    'enlarged_thyroid': 6,
    'brittle_nails': 5,
    'swollen_extremeties': 5,
    'excessive_hunger': 4,
    'extra_marital_contacts': 5,
    'drying_and_tingling_lips': 4,
    'slurred_speech': 4,
    'knee_pain': 3,
    'hip_joint_pain': 2,
    'muscle_weakness': 2,
    'stiff_neck': 4,
    'swelling_joints': 5,
    'movement_stiffness': 5,
    'spinning_movements': 6,
    'loss_of_balance': 4,
    'unsteadiness': 4,
    'weakness_of_one_body_side': 4,
    'loss_of_smell': 3,
    'bladder_discomfort': 4,
    'foul_smell_ofurine': 5,
    'continuous_feel_of_urine': 6,
    'passage_of_gases': 5,
    'internal_itching': 4,
    'toxic_look_(typhos)': 5,
    'depression': 3,
    'irritability': 2,
    'muscle_pain': 2,
    'altered_sensorium': 2,
    'red_spots_over_body': 3,
    'belly_pain': 4,
    'abnormal_menstruation': 6,
    'dischromic_patches': 6,
    'watering_from_eyes': 4,
    'increased_appetite': 5,
    'polyuria': 4,
    'family_history': 5,
    'mucoid_sputum': 4,
    'rusty_sputum': 4,
    'lack_of_concentration': 3,
    'visual_disturbances': 3,
    'receiving_blood_transfusion': 5,
    'receiving_unsterile_injections': 2,
    'coma': 7,
    'stomach_bleeding': 6,
    'distention_of_abdomen': 4,
    'history_of_alcohol_consumption': 5,
    'blood_in_sputum': 5,
    'prominent_veins_on_calf': 6,
    'palpitations': 4,
    'painful_walking': 2,
    'pus_filled_pimples': 2,
    'blackheads': 2,
    'scurring': 2,
    'skin_peeling': 3,
    'silver_like_dusting': 2,
    'small_dents_in_nails': 2,
    'inflammatory_nails': 2,
    'blister': 4,
    'red_sore_around_nose': 2,
    'yellow_crust_ooze': 3
}

description_list = {}
precautionDictionary = {}
symptoms_dict = {}

def getDescription():
    global description_list
    with open('C:\\Users\\Hari\\Downloads\\healthcare-chatbot-master (1)\\healthcare-chatbot-master\\MasterData\\symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            description_list[row[0]] = row[1]

def getprecautionDict():
    global precautionDictionary
    with open('C:\\Users\\Hari\\Downloads\\healthcare-chatbot-master (1)\\healthcare-chatbot-master\\MasterData\\symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            precautionDictionary[row[0]] = row[1:5]

def getInfo():
    print("-----------------------------------HealthCare ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t", end="->")
    name = input("")
    print(f"Hello, {name}")

def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []

def sec_predict(symptoms_exp):
    df = pd.read_csv('C:\\Users\\Hari\\Downloads\\healthcare-chatbot-master (1)\\healthcare-chatbot-master\\Data\\Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X.columns)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[symptoms_dict[item]] = 1

    return rf_clf.predict([input_vector])[0]

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("\nEnter the symptom you are experiencing  \t\t", end="->")
        disease_input = input("")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break

    while True:
        try:
            num_days = int(input("Okay. From how many days? : "))
            break
        except:
            print("Enter valid input.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            symptoms_given = list(cols[reduced_data.loc[present_disease].values[0].nonzero()])
            print("Are you experiencing any ")
            symptoms_exp = []
            for syms in symptoms_given:
                inp = ""
                print(f"{syms}? : ", end='')
                while True:
                    inp = input("")
                    if inp in ["yes", "no"]:
                        break
                    else:
                        print("Provide proper answers i.e. (yes/no) : ", end="")
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            if present_disease[0] == second_prediction:
                print(f"You may have {present_disease[0]}")
                print(description_list[present_disease[0]])
            else:
                print(f"You may have {present_disease[0]} or {second_prediction}")
                print(description_list[present_disease[0]])
                print(description_list[second_prediction])

            precaution_list = precautionDictionary[present_disease[0]]
            print("Take following measures: ")
            for i, j in enumerate(precaution_list):
                print(f"{i + 1}) {j}")

    recurse(0, 1)

# Load data
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf, cols)
print("----------------------------------------------------------------------------------------")

# Tkinter GUI
def on_submit():
    symptoms = symptoms_entry.get("1.0", tk.END).strip().split("\n")
    second_prediction = sec_predict(symptoms)
    output_text.insert(tk.END, f"Symptoms provided: {', '.join(symptoms)}\n")
    output_text.insert(tk.END, f"Primary Prediction: {print_disease(clf.predict([symptoms]))[0]}\n")
    output_text.insert(tk.END, f"Secondary Prediction: {second_prediction}\n")
    output_text.insert(tk.END, f"Description: {description_list[print_disease(clf.predict([symptoms]))[0]]}\n\n")

root = tk.Tk()
root.title("Healthcare ChatBot")

symptoms_label = tk.Label(root, text="Enter your symptoms (one per line):")
symptoms_label.pack()

symptoms_entry = scrolledtext.ScrolledText(root, width=40, height=10)
symptoms_entry.pack()

submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack()

output_label = tk.Label(root, text="ChatBot Output:")
output_label.pack()

output_text = scrolledtext.ScrolledText(root, width=60, height=20)
output_text.pack()

root.mainloop()
