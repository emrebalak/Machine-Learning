import numpy as np
import matplotlib.pyplot as plt

f = open("classification_train.txt")
# To read first line of txt
f.readline()

# Define List for feat1 & feat2 & label
feat1, feat2, label = [], [], []
# Defining zero and one label arrays
zero_label, one_label = [], []
# Search for all line(features) in train data
for line in f:
    splitted_line = line.strip('\n').split("\t")
    # If the line is empty, skip for next one
    if not splitted_line[0]:
        continue

    feat1.append(splitted_line[0])
    feat2.append(splitted_line[1])
    label.append(splitted_line[2])
    if (splitted_line[2] == "1"):
        one_label.append([splitted_line[0], splitted_line[1]])
    else:
        zero_label.append([splitted_line[0], splitted_line[1]])


feat1 = np.array(feat1, dtype=np.float64)
feat2 = np.array(feat2, dtype=np.float64)
label = np.array(label, dtype=np.float64)
zero_label = np.array(zero_label, dtype=np.float64)
one_label = np.array(one_label, dtype=np.float64)

#Finding PC1 and PC2
print(np.size(zero_label,0)/(np.size(zero_label,0)+np.size(one_label,0)))
print(np.size(one_label,0)/(np.size(zero_label,0)+np.size(one_label,0)))

mean_value0 = (np.sum(zero_label,axis=0)/np.size(zero_label,0)) # sum of colums/feature size for label 0
mean_value1 = (np.sum(one_label,axis=0)/np.size(one_label,0)) # sum of colums/feature size for labe 1

#print(mean_value0)
#print(mean_value1)

#Findin Covariance Matrixes
covariance_value0 = np.divide(np.dot(zero_label.transpose(), (zero_label - mean_value0)), np.size(zero_label,0))
covariance_value1 = np.divide(np.dot(one_label.transpose(), (one_label - mean_value1)), np.size(one_label,0))

print(covariance_value0)
print(covariance_value1)

plt.scatter(feat1, feat2)
plt.xlabel("Feat1", fontsize = 16)
plt.ylabel("Feat2", fontsize = 16)

plt.savefig("classification_train.png")