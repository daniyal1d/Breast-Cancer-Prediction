import pickle
import numpy as np

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example input data
input_data = (18.25, 19.98, 119.6, 1040, 0.09463, 0.109, 0.1127, 0.074, 0.1794, 0.05742,
              0.4467, 0.7732, 3.18, 53.91, 0.004314, 0.01382, 0.02254, 0.01039, 0.01369, 0.002179,
              22.88, 27.66, 153.2, 1606, 0.1442, 0.2576, 0.3784, 0.1932, 0.3063, 0.08368)

# Convert input data to numpy array and reshape
input_data_as_numpy_array = np.array(input_data).reshape(1, -1)

# Make prediction
prediction = model.predict(input_data_as_numpy_array)

# Output result
if prediction[0] == 0:
    print('The breast cancer is Malignant')
else:
    print('The breast cancer is Benign')
