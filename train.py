import numpy as np 
import os 
from sklearn.model_selection import train_test_split #to split data into training and testing sets
from tensorflow.keras.utils import to_categorical #to convert labels to one-hot encoding (ohhhhh got it so as an exemple it takes a list of names [apple,orange,idk] and it catogirize them into apple has its own box [1,0,0] and same for the orange [0,1,0] and the idk and then it combine them into a matrix )
from tensorflow.keras.models import Sequential #to build the model (it uses a sequence of layers means neural neural network pattern recognition (input layer, hidden layers, output layer  ))
#LSTM (Long Short-Term Memory)
#Dense = Logic & Decision 
#TensorFlow is an end-to-end open-source platform for machine learning (ML)
from tensorflow.keras.layers import LSTM, Dense 

#use same name of actions in collection.py 
actions =np.array(["hello", "thanks", "iloveyou"])

data_path = "MP_Data"
nb_sequences = 30 #number of frames to collect for each action 
frames = 30 #number of frames to collect for each action 
label_map = {label:num for num, label in enumerate(actions)} #this is a dictionary that maps each action to a number (hello:0, thanks:1, iloveyou:2) 

sequences, labels = [], []
for action in actions:
    for sequence in range(1, nb_sequences+1):
        try:
            window = [] # a list that will have the landmarks x,y,z for 30 frames 
            for frame in range(frames):
                res = np.load(os.path.join(data_path, action, str(sequence), f"{frame}.npy"))
                window.append(res)#as i said we will add the res into window 
            sequences.append(window)# we will add window into the list of sequences 
            labels.append(label_map[action])
        except:
            pass

X = np.array(sequences)
y = to_categorical(labels).astype(int)
