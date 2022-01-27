import matplotlib.pyplot as plot
import numpy as np
import sys
import json
from collections import Counter

data = {}

print(sys.argv[1])

f = open(f"../dataset/{sys.argv[1]}.json", "r")
data = json.loads(f.read())
f.close()

print(data)

num_neurones = [item for sublist in data['activity_stream'] for item in sublist]
num_neurones = list(Counter(num_neurones).keys())
stream = []

print(num_neurones)
print(len(num_neurones))

# neuralData = {}

# for step in range(len(data['activity_stream'])):
#     pass

# Set the random seed for data generation
np.random.seed(2)

# Create rows of random data with 50 data points simulating rows of spike trains
neuralData = []

print(neuralData)

# Set different colors for each neuron
# colorCodes = np.array(
#     [
#         [0, 0, 0],
#         [1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1],
#         [1, 1, 0],
#         [1, 0, 1],
#     ]
# )


# Set spike colors for each neuron
# lineSize = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

plot.yticks(ticks=np.arange(0, len(num_neurones), 1), labels=num_neurones)

# Draw a spike raster plot
plot.eventplot(neuralData, color=[0,0,0], linelengths=0.9)

# Provide the title for the spike raster plot
plot.title("Spike raster plot")

# Give x axis label for the spike raster plot
plot.xlabel("Time (s)")

# Give y axis label for the spike raster plot
plot.ylabel("Spike")

# Display the spike raster plot
plot.show()
