# Data will be split into 80/20
# To keep things proportional, I will split it per target and make it proportional

TARGET = -1 #This is the index of side_b

def split_dataset(data):
    train = []
    test = []

    side_b_groups = {}

    for row in data:
        if row[TARGET] in side_b_groups:
            side_b_groups[row[TARGET]].append(row)
        else:
            side_b_groups[row[TARGET]] = [row]

    for group in side_b_groups.values():
        split = int(len(group) * 0.8)
        train.extend(group[:split])
        test.extend(group[split:])

    return train,test