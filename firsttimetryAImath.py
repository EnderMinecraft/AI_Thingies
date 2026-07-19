import math
import csv
input_data_train=[]
target_data_train=[]
with open('dataset.csv', 'r') as file:
    reader = csv.DictReader(file)
    training_data = []
    for row in reader:
        input_data_train.append([float(row['study_hours_per_week']), float(row['attendance_pct'])])
        target_data_train.append(float(row['pass']))

num_samples=len(input_data_train)
#training_data = [
#    {"input": [0.1, 0.2], "target": 0},
#    {"input": [0.9, 0.6], "target": 1},
#    {"input": [0.9, 0.9], "target": 1},
#    {"input": [0.2, 0.9], "target": 0},
#]

#weight=[4.298762319199204, 4.2987639965956275]
#bias=-6.548212935635646
weight=[2.0223531689321725, 8.275050823568314]
bias= -5.418536189083883
z=0
epoch=0
learning_rate = 0.5

def sigmoid(z):
    return 1/(1+math.e**-z)



def train(epoch):
    global weight, bias
    
    for epoch in range(1, epoch+1):
        tloss=0
        tdw1=0
        tdw2=0
        tdbias=0
        for i in range(num_samples):
            input_data = input_data_train[i]
            target_data = target_data_train[i]
            z=(input_data[0]*weight[0])+(input_data[1]*weight[1])+bias
            z_sigmoid=sigmoid(z)
            
            loss = 1/2 * (z_sigmoid - target_data)**2
            tloss += loss
            
            dloss=z_sigmoid-target_data
            filter=z_sigmoid*(1-z_sigmoid)
            basedelta = dloss*filter
            
            tdw1=tdw1 + basedelta*input_data[0]
            tdw2=tdw2 + basedelta*input_data[1]
            tdbias=tdbias + basedelta*1
        weight[0]=weight[0]-learning_rate*(tdw1/num_samples)
        weight[1]=weight[1]-learning_rate*(tdw2/num_samples)
        bias=bias-learning_rate*(tdbias/num_samples)
        if epoch%100==0:
            print("Epoch: ",epoch," Avg. Loss: ",tloss/num_samples)
    print("Weight: ",weight," Bias: ",bias)



def testmodel(x1,x2):
    z=(x1*weight[0])+(x2*weight[1])+bias
    prob=sigmoid(z)
    print(f"Input: ({x1}, {x2}) | Probability: {prob*100:.4f} % | Prediction: {1 if prob > 0.5 else 0}")

i=input("Sel mode\n 1. Train\n 2.Test model\n 3. Test with sample data\n")
if i=="1":
    train(epoch=int(input("Epoch: ")))
if i=="2":
    print(" Study time (hrs/week), Attend rate (%)")
    input1=float(input("Input 1: "))/100
    input2=float(input("Input 2: "))/100
    testmodel(input1, input2)
if i=="3":
    testmodel(12/100,90/100)
    testmodel(15/100,50/100)
    testmodel(3/100,95/100)
    testmodel(6/100,70/100)
    testmodel(2/100,30/100)