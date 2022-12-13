# %%
from sklearn.tree import DecisionTreeClassifier
import pickle

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

# packaged the model
with open('classifier.pkl', 'wb') as file:
    pickle.dump(clf, file)



