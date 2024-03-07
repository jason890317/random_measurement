import sys

def print_progress(current, total, bar_length=20):
    """
    Prints a progress bar to the console.
    
    Parameters:
        current (int): The current progress value.
        total (int): The total value when the progress is complete.
        bar_length (int): The character length of the progress bar.
    """
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '=' + '>'
    padding = (bar_length - len(arrow)) * ' '
    progress_bar = f'\r{current}/{total}: [{arrow}{padding}]'
    sys.stdout.write(progress_bar)
    sys.stdout.flush()

# Example usage
import time

total = 100
for i in range(1, total + 1):
    print_progress(i, total, 20)
    time.sleep(0.05)  # Simulate work being done

print()  # Move to the next line
