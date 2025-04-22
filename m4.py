import csv

num_attributes = 6
a = []

print("\nThe Given Training Data Set\n")
with open('training_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a.append(row)
        print(row)

print("\nThe initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)

for j in range(0, num_attributes):
    hypothesis[j] = a[0][j]

print("\nFind-S: Finding a Maximally Specific Hypothesis\n")
for i in range(0, len(a)):
    if a[i][num_attributes] == 'Yes':
        for j in range(0, num_attributes):
            print(a[i][j], end=' ')
            if a[i][j] != hypothesis[j]:
                hypothesis[j] = '?'
            else:
                hypothesis[j] = a[i][j]
        print("\n\nFor Training instance No:{} the hypothesis is ".format(i), hypothesis)

print("\nThe Maximally Specific Hypothesis for a given Training Examples:\n")
print(hypothesis)


''' 
training_data.csv
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes
'''
