import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


# Define the symptoms and possible values
symptoms = ['coughing', 'sneezing', 'vomiting', 'diarrhea', 'loss of appetite', 'lethargy', 'fever', 'swelling', 'lumps'
            'persistent sore', 'change in appetite', 'dehydration', 'weight loss', 'skin infection', 'nausea', 'foamy mouth',
            'hypersensitive', 'restleness', 'agression', 'listleness', 'bad breath', 'refuse eat dryfood', 'mouth blood discharge',
            'ear canal redness', 'scabs around ears', 'hairloss around ear', 'balance issues', 'anemia', 'gingivitis', 'stomatitis',
            'enlarge lymph nodes', 'jaundice', 'abcesses', 'losse teeth']

# Define the diseases and possible values
diseases = ['no symtomps','kennel cough', 'allergies', 'parvovirus', 'giardiasis', 'feline leukemia', 'feline immunodeficiency virus',
            'cancer', 'diabetes', 'heartworm', 'dental', 'rabies', 'ear infection', 'FIV', 'FeIV']
disease_values = [0, 1, 2, 3, 4, 5]

# Define the training data
# Format: [symptoms, disease]
training_data = [
    # sequence template [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0],   # no symptoms
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1],  #  kennel cough
    [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 2],  #  allergies
    [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3],  #  parvovirus
    [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 4],  #  giardiasis
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 5],  #  leukemia
    [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 6],  #  feline immunodeficiency virus
    [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 7],  #  cancer
    [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8],  #  diabetes
    [[1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 9],    #   heartworm
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 10],    #   dental infection
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 11],   #   rabies
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 12],   #   ear infection
    [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], 13],   #   feline immunodeficiency virus
    [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], 14],   #   feline immunodeficiency virus
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 11],   #   rabies
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8],  #  diabetes
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8],  #  diabetes
]

# symptoms = ['0coughing', '1sneezing', '2vomiting', '3diarrhea', '4loss of appetite', '5lethargy', '6fever', '7swelling', '8lumps'
#             '9persistent sore', '10change in appetite', '11dehydration', '12weight loss', '13skin infection', '14nausea', '15foamy mouth',
#             '16hypersensitive', '17restleness', '18agression', '19listleness', '20bad breath', '21refuse eat dryfood', '22mouth blood discharge',
#             '23ear canal redness', '24scabs around ears', '25hairloss around ear', '26balance issues', '27anemia', '28gingivitis', '29stomatitis',
#             '30enlarge lymph nodes', '31jaundice', '32abcesses', '33losse teeth']

# Split 
X = [row[0] for row in training_data]
y = [row[1] for row in training_data]

# instance
classifier = DecisionTreeClassifier()

# Train mo splitted data
clf = classifier.fit(X, y)



###########################################################################################

st.write("""Pet Disease Prediction App""")
st.sidebar.header('Symtomps Parameter')

def user_input():
    cough = st.sidebar.checkbox('Cough', value=False),
    sneeze = st.sidebar.checkbox('Sneeze', value=False),
    vomit = st.sidebar.checkbox('Vomit', value=False),
    diarrhea = st.sidebar.checkbox('Diarrhea', value=False),
    loss_of_appetite = st.sidebar.checkbox('Loss of appetite', value=False),
    lethargy = st.sidebar.checkbox('Lethargy', value=False),
    fever = st.sidebar.checkbox('Fever', value=False),
    swelling = st.sidebar.checkbox('Swelling', value=False),
    lumps = st.sidebar.checkbox('Lumps', value=False),
    persistent_sore = st.sidebar.checkbox('Persistent sore', value=False),
    change_in_appetite = st.sidebar.checkbox('Change in appetite', value=False),
    dehydration = st.sidebar.checkbox('Dehydration', value=False),
    weight_loss = st.sidebar.checkbox('Weight loss', value=False),
    skin_infection = st.sidebar.checkbox('Skin infection', value=False),
    nausea = st.sidebar.checkbox('Nausea', value=False),
    foamy_mouth = st.sidebar.checkbox('Foamy mouth', value=False),
    hypersensitive = st.sidebar.checkbox('Hyper sensitive', value=False),
    restleness = st.sidebar.checkbox('Restlesness', value=False),
    aggression = st.sidebar.checkbox('Aggression', value=False),
    listleness = st.sidebar.checkbox('Listleness', value=False),
    bad_breath = st.sidebar.checkbox('Bad breath', value=False),
    refuse_eat_dryfood = st.sidebar.checkbox('Refuse eat dryfood', value=False),
    mouth_blood_discharge = st.sidebar.checkbox('Mouth blood discharge', value=False),
    ear_canal_redness = st.sidebar.checkbox('Ear canal redness', value=False),
    scabs_around_ear = st.sidebar.checkbox('Scabs around ears', value=False),
    hairloss_around_ear = st.sidebar.checkbox('Hair loss around ear', value=False),
    balance_issue = st.sidebar.checkbox('Balance issue', value=False),
    anemia = st.sidebar.checkbox('Anemia', value=False),
    gingivitis = st.sidebar.checkbox('Gingivitis', value=False),
    stomatitis =st.sidebar.checkbox('Stomatitis', value=False)
    enlarge_lymph_nodes = st.sidebar.checkbox('Enlarge lymph nodes', value=False),
    jaundice = st.sidebar.checkbox('Jaundice', value=False),
    abscesses = st.sidebar.checkbox('Abscesses', value=False),
    loose_teeth = st.sidebar.checkbox('Loose_teeth', value=False),

    data = {
        'cough': cough,
        'sneeze': sneeze,
        'vomit': vomit,
        'diarrhea': diarrhea,
        'loss of appetite': loss_of_appetite,
        'lethargy': lethargy,
        'fever': fever,
        'swelling': swelling,
        'lumps': lumps,
        'persistent sore': persistent_sore,
        'change in appetite': change_in_appetite,
        'dehydration': dehydration,
        'weight loss': weight_loss,
        'skin infection': skin_infection,
        'nausea': nausea,
        'foamy mouth': foamy_mouth,
        'hypersensitive': hypersensitive,
        'restleness': restleness,
        'aggression': aggression,
        'listleness': listleness,
        'bad breath': bad_breath,
        'refuse eat dryfood': refuse_eat_dryfood,
        'mouth blood discharge': mouth_blood_discharge,
        'ear canal redness': ear_canal_redness,
        'scabs around ear': scabs_around_ear,
        'hairloss around ear': hairloss_around_ear,
        'balance issue': balance_issue,
        'anemia': anemia,
        'gingivitis': gingivitis,
        'stomatitis': stomatitis,
        'enlarge lymph nodes': enlarge_lymph_nodes,
        'jaundice': jaundice,
        'abscesses': abscesses,
        'loose teeth': loose_teeth
    }

    features = pd.DataFrame(data, index=[1])
    
    return features

df = user_input()

st.subheader('User Input Parameters')
st.write(df)

# Use the classifier in your Streamlit app
prediction = clf.predict(df)

st.subheader("""Possible disease""")


if prediction == 0:
    st.write('no prediction')
elif prediction == 1:
    st.write('**Kennel cough**')
elif prediction == 2:
    st.write('**Allergies**')
elif prediction == 3:
    st.write('**Parvo Virus**')
elif prediction == 4:
    st.write('**Giardiasis**')
elif prediction == 5:
    st.write('**Leukemia**')
elif prediction == 6:
    st.write('**Immuno deficiency virus**')
elif prediction == 7:
    st.write('**Cancer**')
elif prediction == 8:
    st.write('**Diabetes**')
elif prediction == 9:
    st.write('**Heartworm**')
elif prediction == 10:
    st.write('**Dental Infection**')
elif prediction == 11:
    st.write('**Rabies**')
elif prediction == 12:
    st.write('**Ear infection**')
elif prediction == 13:
    st.write('**Immuno deficiency virus**')
elif prediction == 14:
    st.write('**Immuno deficiency virus**')

