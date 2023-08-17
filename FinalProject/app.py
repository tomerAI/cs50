from cs50 import SQL
import pandas as pd
from flask import Flask, flash, render_template, request
import pickle
import numpy as np
import csv

# Configure application
app = Flask(__name__)
app.secret_key = "secret key"

# Open model
with open('gb-model.pkl', 'rb') as f:
    model = pickle.load(f)

# Categorical feature values
with open('feature_columns', 'rb') as f:
    features = pd.read_csv(f)
    homeplanets = features['HomePlanet'].unique()
    cryosleeps = features['CryoSleep'].unique()
    destinations = features['Destination'].unique()
    vips = features['VIP'].unique()
    decks = features['Deck'].unique()
    sides = features['Side'].unique()

# Sorting lists
features_list = [homeplanets, cryosleeps, destinations, vips, decks, sides]
for feature in features_list:
     feature.sort()

# Feature names
#with open('feature_names.csv', 'r') as f:
    #features_names_list = list(csv.reader(f, delimiter=","))

# Configure SQLite database
db = SQL("sqlite:///history.db")


@app.route("/", methods=['POST', 'GET'])
def index():
    """Predict survival on the spaceship titanic by a passenger chosen by user"""
    history = db.execute("SELECT * FROM HISTORY ORDER BY ID DESC")

    # Handles POST request
    if request.method == "POST":
            # Handles category requests
            homeplanet = request.form.get('homeplanet')
            cryosleep = request.form.get('cryosleep')
            destination = request.form.get('destination')
            vip = request.form.get('vip')
            deck = request.form.get('deck')
            side = request.form.get('side')
            categories = [homeplanet, cryosleep, destination, vip, deck, side]
            for category in categories:
                if category is None:
                    flash("Fill out missing categories")
                    return render_template("predict.html", homeplanets=homeplanets, cryosleeps=cryosleeps, destinations=destinations, 
                                   vips=vips, decks=decks, sides=sides, history=history)
                

            # Handles numerical requests
            age = request.form.get("age")
            roomservice = request.form.get("roomservice")
            foodcourt = request.form.get("foodcourt")
            shoppingmall = request.form.get("shoppingmall")
            spa = request.form.get("spa")
            vrdeck = request.form.get("vrdeck")
            numbers = [age, roomservice, foodcourt, shoppingmall, spa, vrdeck]
            int_numbers = []
            for number in numbers:
                 if not number.isdigit() or int(number) < 0:
                    flash("Fill out numbers")
                    return render_template("predict.html", homeplanets=homeplanets, cryosleeps=cryosleeps, destinations=destinations, 
                                   vips=vips, decks=decks, sides=sides, history=history) 
                 else:
                     number = int(number)
                     int_numbers.append(number)
            
             

            # Creating array for features
            X = []
            # integers
            for number in int_numbers:
                X.append(number)

            def one_hot_encoding_bool(array: list, value) -> list:
                for _ in array:
                    _ = str(_)
                    if _ == value:
                        _ = True
                        X.append(_)
                    else:
                        _ = False
                        X.append(_)
            
            # Initializing values
            x = 0
            # One-hot_encode categories and bools
            for i in features_list:
                one_hot_encoding_bool(i, categories[x])
                x += 1

            # Turn list into 1D array
            features_values = np.array(X).reshape(1, 28)

            # Predict model using inputs
            prediction = model.predict_proba(features_values)

            #print(prediction)
            if prediction[0][0] > 0.5:
                transported = "False"
                percentage = prediction[0][0] * 100
                statement = f"Your passenger stayed on the Spaceship Titanic with {percentage: .2f}% likelihood"
            if prediction[0][0] < 0.5:
                transported = "True"
                percentage = prediction[0][1] * 100
                statement = f"Your passenger was transported to an alternate dimension with {percentage: .2f}% likelihood"

            transported_percentage = round(prediction[0][1] * 100, 2)

            db.execute("INSERT INTO HISTORY (HOMEPLANET, CRYOSLEEP, DESTINATION, VIP, DECK, SIDE, AGE, ROOMSERVICE, FOODCOURT, SHOPPINGMALL, SPA, VRDECK, TRANSPORTED, TRANSPORTED_PERCENTAGE) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", homeplanet, cryosleep, destination, vip, deck, side, age, roomservice, foodcourt, shoppingmall, spa, vrdeck, transported, transported_percentage)

            history = db.execute("SELECT * FROM HISTORY ORDER BY ID DESC")

            return render_template("predict.html", statement=statement, transported=transported, transported_percentage=transported_percentage, homeplanets=homeplanets, cryosleeps=cryosleeps, destinations=destinations, 
                                   vips=vips, decks=decks, sides=sides, history=history)
    
    # Handles GET request
    else:
        return render_template("predict.html", homeplanets=homeplanets, cryosleeps=cryosleeps, destinations=destinations, 
                                   vips=vips, decks=decks, sides=sides, history=history)
    
