#Get data and preprocess function

import csv;
from datetime import datetime

#Gets Location (Longitude, Latitude)
#Gets Lethality (deaths_a,deaths_b,deaths_civilians,best)
#Gets type_of_violence and duration_days

#Gets side_b (This is Y or the target)

#Constants for indexing

# Target
SIDE_B = 17

# Location
LATITUDE = 29
LONGITUDE = 30

# Lethality
DEATHS_A = 41
DEATHS_B = 42
DEATHS_CIVILIANS = 43
BEST = 45

# Type of Violence & Duration
TYPE_OF_VIOLENCE = 5
DATE_START = 39  # Used to compute duration_days
DATE_END = 40  # Used to compute duration_days

# Excluded data
EXCLUDED = ["Civilians","Government of Kuwait","Military faction (forces of Honasan, Abenina & Zumel)"]

TARGET_IDX = -1

def load_data(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        return list(reader)

def preprocess(csv_file):
    raw_data = load_data(csv_file)
    preprocessed = []
    for row in raw_data[1:]: #[LONGITUDE,LATITUDE,DEATHS_A,DEATHS_B,DEATHS_CIVILIANS,BEST,TYPE_OF_VIOLENCE,DURATION_DAYS,SIDE_B]
        #Exclude Civilians since it is already in the input, and other unrelated to Insurgency and Mindanao
        if row[SIDE_B] in EXCLUDED:
            continue
        #Process date first
        duration = 0
        if row[DATE_START] != "" and row[DATE_END] != "":
            date_start = datetime.fromisoformat(row[DATE_START])
            date_end = datetime.fromisoformat(row[DATE_END])
            duration = (date_end - date_start).days
        preprocessed.append([float(row[LONGITUDE]),float(row[LATITUDE]),int(row[DEATHS_A]),int(row[DEATHS_B]),int(row[DEATHS_CIVILIANS]),int(row[BEST]),int(row[TYPE_OF_VIOLENCE]),duration,row[SIDE_B]])

    return preprocessed

def classify_groups(data): #Changes Insurgency to 0 and others to 1, which is the mindanao groups
    for row in data:
        if row[TARGET_IDX] == "CPP":
            row[TARGET_IDX] = 0
        else:
            row[TARGET_IDX] = 1
    return data