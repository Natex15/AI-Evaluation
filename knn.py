# Will use 2 for minkowski distance

class KNN:
    def __init__(self, k=5, p=2):
        self.k = k
        self.p = p

    def fit(self, training_data):
        self.training_data = training_data

    def predict(self, testing_data): #[LONGITUDE,LATITUDE,DEATHS_A,DEATHS_B,DEATHS_CIVILIANS,BEST,TYPE_OF_VIOLENCE,DURATION_DAYS,SIDE_B]
        #Index for features
        LONGITUDE = 0
        LATITUDE = 1
        DEATHS_A = 2
        DEATHS_B = 3
        DEATHS_CIVILIANS = 4
        BEST = 5
        TYPE_OF_VIOLENCE = 6
        DURATION_DAYS = 7
        SIDE_B = 8

        result = [] #[[prediction,correct answer],..]

        for test in testing_data: #i is for record number
            distances = []
            for train in self.training_data: #should append (distance,classification)
                d = (abs(float(test[LONGITUDE])-float(train[LONGITUDE]))**self.p+abs(float(test[LATITUDE])-float(train[LATITUDE]))**self.p+abs(float(test[DEATHS_A])-float(train[DEATHS_A]))**self.p+abs(float(test[DEATHS_B])-float(train[DEATHS_B]))**self.p+abs(float(test[DEATHS_CIVILIANS])-float(train[DEATHS_CIVILIANS]))**self.p+abs(float(test[BEST])-float(train[BEST]))**self.p+abs(float(test[TYPE_OF_VIOLENCE])-float(train[TYPE_OF_VIOLENCE]))**self.p+abs(float(test[DURATION_DAYS])-float(train[DURATION_DAYS]))**self.p)**(1/self.p)
                distances.append((d,train[SIDE_B]))
            distances.sort()
            distances = distances[:self.k]
            
            counts = self.counter(distances)
            # Check for equal max counts, if max, pops until not equal, meaning the closest is the one that is picked
            while (max(counts) == counts[0] and max(counts) == counts[1]) or (max(counts) == counts[1] and max(counts) == counts[2]) or (max(counts) == counts[0] and max(counts) == counts[2]):
                distances.pop()
                counts = self.counter(distances)
            prediction = counts.index(max(counts))
            result.append(prediction)
        return result
            
    def counter(self, distances): # Returns the count list
        counts = [0,0,0]
        for distance in distances:
            counts[int(distance[1])]+=1
        return counts