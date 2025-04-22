import csv

num_attributes = 6

print("\nThe Given Training Data Set\n")
with open('training_data.csv', 'r') as csvfile:
    data = list(csv.reader(csvfile))
    for row in data:
        print(row)
    
hypothesis = data[0][:num_attributes]
print("\nThe initial value of hypothesis:")
print(['0'] * num_attributes)

print("\nFind S: Finding a Maximally Specific Hypothesis\n")
for i, row in enumerate(data):
    if row[-1] == 'Yes':
        for j in range(num_attributes):
            print(row[j], end=' ')
            if row[j] != hypothesis[j]:
                hypothesis[j] = '?'
    print(f"\n\nFor Training instance No:{i} the hypothesis is", hypothesis)

print("\nThe Maximally Specific Hypothesis for the given Training Examples:\n")
print(hypothesis)

/*training_data.csv
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes
*/
