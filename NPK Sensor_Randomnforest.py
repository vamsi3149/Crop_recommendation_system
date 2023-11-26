import serial
import time

# Open a serial connection to your Arduino. Adjust the port and baud rate as needed.
ser = serial.Serial('COM6', 9600)  # Replace 'COM3' with the correct serial port

try:
    i=0
    while True:
        data = ser.readline().decode().strip()
        i+=1
        if i==4:
            N=data
        elif i==5:
            P=data
        elif i==6:
            K=data
        elif i==7:
            temp=data
        elif i==8:
            hum=data
        elif i==9:
            break   

except KeyboardInterrupt:
    ser.close()  # Close the serial connection when done

except serial.SerialException:
    print("Serial connection error. Make sure the Arduino is connected and the port is correct.")
print(N,P,K,temp,hum)



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Split the data into features (temperature, humidity, N, P, K) and target (recommended crop)
X = data[['temperature', 'humidity', 'N', 'P', 'K']]
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
clf.fit(X_train, y_train)

# Now, you can use the trained model to recommend the best crop based on input data
# Replace 'new_data' with the actual data (Temperature, Humidity, N, P, K) you obtain
new_data = pd.DataFrame({'temperature': [temp], 'humidity': [hum], 'N': [N], 'P': [P], 'K': [K]})

recommended_crop = clf.predict(new_data)

# Print the input values column-wise
for col in new_data.columns:
    print(f"{col}: {new_data[col].values[0]}")

print(f"Recommended Crop: {recommended_crop[0]}")
