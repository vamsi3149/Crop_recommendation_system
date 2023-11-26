import tkinter as tk
import serial
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import recall_score


# Function to update the displayed data and recommendation
def update_display():
    global nitrogen, phosphorous, potassium, temperature, humidity, last_update_time
    try:
        while True:
            data = ser.readline().decode('utf-8').strip()
            if data.startswith("Nitrogen:"):
                nitrogen = float(data.split(" ")[1])
            elif data.startswith("Phosphorous:"):
                phosphorous = float(data.split(" ")[1])
            elif data.startswith("Potassium:"):
                potassium = float(data.split(" ")[1])
            elif data.startswith("Temperature:"):
                temperature = float(data.split(" ")[1].strip("°C"))
            elif data.startswith("Humidity:"):
                humidity = float(data.split(" ")[1].strip("%"))
                break  # Exit the loop after extracting all the data

        new_data = pd.DataFrame({'temperature': [temperature], 'humidity': [humidity], 'N': [nitrogen], 'P': [phosphorous], 'K': [potassium]})
        recommended_crop = clf.predict(new_data)
        y_true = [recommended_crop[0]]  # Assuming that 'label' is the ground truth
        y_pred = clf.predict(new_data)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')

        # Update the displayed values
        nitrogen_label.config(text="Nitrogen: {:.2f} mg/kg".format(nitrogen))
        phosphorous_label.config(text="Phosphorous: {:.2f} mg/kg".format(phosphorous))
        potassium_label.config(text="Potassium: {:.2f} mg/kg".format(potassium))
        temperature_label.config(text="Temperature: {:.2f} °C".format(temperature))
        humidity_label.config(text="Humidity: {:.2f}%".format(humidity))

        # Update the recommended crop representation
        recommendation_label.config(text="Recommended Crop: {}".format(recommended_crop[0]))

        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        if current_time != last_update_time:
            last_update_time = current_time
            time_label.config(text="Last Update: {}".format(current_time))

        update_graphs()
        root.after(1000, update_display)  # Schedule the next update after 5 seconds
    except KeyboardInterrupt:
        ser.close()

# Function to update graphical representations
def update_graphs():
    data = [nitrogen, phosphorous, potassium, temperature, humidity]
    labels = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity']
    styles = ['b-', 'g--', 'r-.', 'c:', 'm-']

    # Define a fixed y-axis range for all graphs
    y_range = (0, 100)  # Adjust the range as needed

    for i, label in enumerate(labels):
        axes[i].cla()
        x = np.arange(0, 1, 0.1)  # Sample x values (you can customize this)
        y = [data[i] for _ in x]  # Use data[i] as y values (change this depending on your data)
        axes[i].plot(x, y, styles[i], label=label)
        axes[i].set_title(label)
        axes[i].set_xlabel('instances')
        axes[i].set_ylabel('Output Values')
        axes[i].set_ylim(y_range)  # Set the y-axis range
        axes[i].legend()

    canvas.draw()

# Initialize the serial port, variables, and the Random Forest classifier
serial_port = 'COM6'  # Change to your Arduino's serial port
ser = serial.Serial(serial_port, 9600)
nitrogen, phosphorous, potassium, temperature, humidity, last_update_time = 0, 0, 0, 0, 0, ""
data = pd.read_csv("E:\python_testing\Crop_recommendation.csv")
X = data[['temperature', 'humidity', 'N', 'P', 'K']]
y = data['label']
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Create the Tkinter GUI
root = tk.Tk()
root.title("Crop Recommendation System")

# Create a custom font for labels
custom_font = ("Ink Free", 20)
background_image = tk.PhotoImage(file="background.png")
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=10, relheight=10)
# Labels to display data and recommendation
nitrogen_label = tk.Label(root, text="Nitrogen: {:.2f} mg/kg", font=custom_font)
nitrogen_label.pack()
phosphorous_label = tk.Label(root, text="Phosphorous: {:.2f} mg/kg", font=custom_font)
phosphorous_label.pack()
potassium_label = tk.Label(root, text="Potassium: {:.2f} mg/kg", font=custom_font)
potassium_label.pack()
temperature_label = tk.Label(root, text="Temperature: {:.2f} °C", font=custom_font)
temperature_label.pack()
humidity_label = tk.Label(root, text="Humidity: {:.2f}%", font=custom_font)
humidity_label.pack()
recommendation_label = tk.Label(root, text="Recommended Crop:", font=custom_font)
recommendation_label.pack()
time_label = tk.Label(root, text="Last Update:", font=custom_font)
time_label.pack()


# Create a figure and axes for graphical representations
figure, axes = plt.subplots(1,5, figsize=(20, 10))
canvas = FigureCanvasTkAgg(figure, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Start updating the display
update_display()

root.mainloop()
