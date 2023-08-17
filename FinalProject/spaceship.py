import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

filepath = './data/train.csv'
df = pd.read_csv(filepath)

""" 
Data description
train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
Destination - The planet the passenger will be debarking to.
Age - The age of the passenger.
VIP - Whether the passenger has paid for special VIP service during the voyage.
RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
Name - The first and last names of the passenger.
Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.


NOTES
Passengers are in a group when pp >= 02 
Cabin must be split between deck/num/side
RoomService, FoodCourt, ShoppingMall, Spa, VRDeck shows moneyspending/status

TASK: Transported is bool --> binary classification


test.csv - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.

sample_submission.csv - A submission file in the correct format.
PassengerId - Id for each passenger in the test set.
Transported - The target. For each passenger, predict either True or False.


To Do
- Fill in null values
- Balance out the dataset if imbalanced
- Drop high cardinality columns (Identifier columns: names, passenger id)

"""




###                 Splitting cabin
def split_cabin(x):
    if len(str(x).split('/')) < 3:
        return ["Missing", "Missing", "Missing"]
    else:
        return str(x).split('/')

###                 Preprocessing dataset
def preprocessing(df) -> df:
    # Homeplanet - Fill NA with missing
    df['HomePlanet'].fillna('Missing', inplace=True)
    # CryoSleep - high corr - drop NA 
    #df['CryoSleep'].dropna()
    # Cabin - Split column D/N/S into Deck and Side - Drop Number
    df['TempCabin'] = df['Cabin'].apply(lambda x: split_cabin(x))
    df['Deck'] = df['TempCabin'].apply(lambda x: x[0])
    df['Side'] = df['TempCabin'].apply(lambda x: x[2])
    df.drop('TempCabin', axis=1, inplace=True)
    # Destination - Fill NA with missing
    df['Destination'].fillna('Missing', inplace=True)
    # Age - Fill NA with mean
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    # Monetary - Set NA to 0 (60% is 0 spending)
    df['RoomService'].fillna(0, inplace=True)
    df['FoodCourt'].fillna(0, inplace=True)
    df['ShoppingMall'].fillna(0, inplace=True)
    df['Spa'].fillna(0, inplace=True)
    df['VRDeck'].fillna(0, inplace=True)
    # VIP - Drop NA
    # Name - Drop due to high cardinality
    df.drop('Name', axis=1, inplace=True)
    # Remaining - Drop NA
    df.dropna(inplace=True)

###                 Analytical Base Table
abt = df.copy()
preprocessing(abt)
#print(abt.info())


###                 Modelling
"""
- Feature and Target values - X, y
- One hot encode any categorical features
- Train, holdout split
- Train on a bunch of algos
"""


###                 Creating features columns - Drop target, identifier and cabin column
X = abt.drop(["Transported", "PassengerId", "Cabin"], axis=1)
#print(X.describe(include="object"))
#print(X.describe(exclude="object"))
#X.to_csv('feature_columns', index=False)
# One-hot encode columns
X = pd.get_dummies(X)
#print(X)
"""

print(X.info())
print(X.columns)
"""

# Create csv file of feature names
with open('feature_names.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(X.columns)

# Create csv file of X
pd.DataFrame(X).to_csv('X_train.csv')

# Creating target column
y = abt['Transported']

# Create csv file of y
pd.DataFrame(y).to_csv('y_train.csv')

###                 Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#print(y_train.head())


###                 Setup ML Pipelines
pipelines = {
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42)),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42)),
}

###                 Setup ML hyperparameter-tuning
grid = {
    'rf': {
        'randomforestclassifier__n_estimators':[100,200,300]
    },
    'gb': {
        'gradientboostingclassifier__n_estimators':[100,200,300]
    }
}

###                 Create models from algo-pipelines and parameter-grids 

fit_models = {}
for algo, pipeline in pipelines.items():
    print(f'Training the {algo} model')
    model = GridSearchCV(pipeline, grid[algo], n_jobs=-1, cv=10)
    model.fit(X_train, y_train)
    fit_models[algo] = model


###                 Evaluate performance of models
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    precision = precision_score(y_test, yhat)
    recall = recall_score(y_test, yhat)
    print(f"Metrics for {algo}: accuracy= {accuracy}, precision= {precision} and recall= {recall}")



"""
Metrics for rf: accuracy= 0.7952844311377245, precision= 0.8143525741029641 and recall= 0.7716186252771619
Metrics for gb: accuracy= 0.8080089820359282, precision= 0.795774647887324 and recall= 0.8351810790835181
"""
# Best performing model is GradientBoostingClassifier


###                 Save gb-model in local file

with open('gb-model.pkl', 'wb') as f:
    pickle.dump(fit_models['gb'], f)


