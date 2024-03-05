import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Example list of numbers, including a very small number
numbers = [1, 2, 3, 4, 1, 2, 1, 3, 5, 6, 5, 5, 6e-5]

# Count the frequency of each number
number_counts = Counter(numbers)

# Separate the numbers and their counts into two lists for plotting
labels, values = zip(*number_counts.items())

# Convert labels to string to handle very small numbers and ensure they appear on the x-axis
labels_str = [f'{label:.1e}' if label < 1e-3 else str(label) for label in labels]

# Create the plot
plt.figure(figsize=(10, 6))  # Adjust the figure size
plt.bar(labels_str, values, color='skyblue')  # Create a bar chart

# Add title and labels to the plot
plt.title('Frequency of Each Number')
plt.xlabel('Number')
plt.ylabel('Frequency')

# Set the y-axis to a logarithmic scale
plt.yscale('log')

# Optional: Set y-axis limits if you want to focus on a specific range
plt.ylim(bottom=1e-4, top=max(values)+10)

# Show the plot
plt.show()
