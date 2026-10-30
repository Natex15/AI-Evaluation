# Data will be split into 80/20
# To keep things proportional, I will split it per target and make it proportional
import random

TARGET = -1 #This is the index of side_b
SEED = 67 #Used for random

def split_dataset(data):
    train = []
    test = []

    side_b_groups = {}

    random.seed(SEED)

    for row in data:
        if row[TARGET] in side_b_groups:
            side_b_groups[row[TARGET]].append(row)
        else:
            side_b_groups[row[TARGET]] = [row]

    for group in side_b_groups.values():
        split = int(len(group) * 0.8)
        random.shuffle(group) #Makes the order random and not tied by chronological order
        train.extend(group[:split])
        test.extend(group[split:])

    #Shuffles again the train and test
    random.shuffle(train)
    random.shuffle(test)

    return train,test