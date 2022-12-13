import streamlit as st
import pandas as pd
import pickle
import numpy as np


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

# Deserialize the classifier object using pickle
with open('classifier.pkl', 'rb') as file:
    clf = pickle.load(file)

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

