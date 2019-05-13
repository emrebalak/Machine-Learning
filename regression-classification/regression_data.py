import numpy as np
import matplotlib.pyplot as plt


f = open("regression_data.txt")
# To read first line of txt
f.readline()

# Define List for Brain Weight & Head Size
X, Y = [], [] 
# Search for all line(features) in train data
for line in f:
	splitted_line = line.strip('\n').split("\t\t\t")
	# If the line is empty, skip for next one
	if not splitted_line[0]:
		continue

	X.append(splitted_line[0])
	Y.append(splitted_line[1])


X = np.array(X, dtype=np.float64)
Y = np.array(Y, dtype=np.float64)

#Creating dots for values
plt.scatter(X, Y)
plt.xlabel("Head Size (cm^3)", fontsize = 16)
plt.ylabel("Brain Weight (grams)", fontsize = 16)


Q0, Q1 = 0, 0
learningRate = 0.000000001 
iterations = 10000

# Performing Gradient Descent 
for i in range(iterations): 
    Y_pred = Q1*X + Q0 
    DQ1 = (-2/float(len(X))) * sum(X * (Y - Y_pred)) 
    DQ0 = (-2/float(len(X))) * sum(Y - Y_pred) 
    Q1 = Q1 - learningRate * DQ1 
    Q0 = Q0 - learningRate * DQ0 

#Plotting the line
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='black')
plt.savefig("regression_data.png")

# Splitting array for 5-fold CV
k = 5
folds = np.array_split(X,k)
folds_y = np.array_split(Y,k)

Q1_list, Q0_list, mse_arr = [], [], []

for i in range(k):
    train = folds.copy()
    predict = folds_y.copy()
    test = folds[i]
    test_y = folds_y[i]
    del train[i]
    del predict[i]
    train_set = np.concatenate(train[:i] + train[i+1:], axis = 0)
    predict_set = np.concatenate(predict[:i] + predict[i+1:], axis = 0)

    Q0, Q1 = 0, 0
    learningRate = 0.000000001 
    iterations = 10000 
    for i in range(iterations): # Performing Gradient Descent
        Y_pred = Q1*train_set + Q0 
        DQ1 = (-2/float(len(X))) * sum(train_set * (predict_set - Y_pred)) 
        DQ0 = (-2/float(len(X))) * sum(predict_set - Y_pred) 
        Q1 = Q1 - learningRate * DQ1 
        Q0 = Q0 - learningRate * DQ0 
        
    Q1_list.append(Q1)
    Q0_list.append(Q0)
    mse = sum(((predict_set - Y_pred)**2))/len(predict_set) #Finding MSE
    mse_arr.append(mse)
    
mse_total = sum(mse_arr)/5 #MSE overall
print(mse_total)
