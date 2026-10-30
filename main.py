import preprocess
import split

#[LONGITUDE,LATITUDE,DEATHS_A,DEATHS_B,DEATHS_CIVILIANS,BEST,TYPE_OF_VIOLENCE,DURATION_DAYS,SIDE_B]
processed_data = preprocess.preprocess('conflict_data_phl.csv')

for row in processed_data:
    print(row)

train, test = split.split_dataset(processed_data)
train = preprocess.classify_groups(train)
test = preprocess.classify_groups(test)

print(len(train))
for row in train:
    print(row)

print(len(test))
for row in test:
    print(row)