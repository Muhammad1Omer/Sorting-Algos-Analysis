import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize the figure once
fig, ax = plt.subplots(figsize=(10, 5))

# Function to draw bars
def draw_bars(arr, step):
    """Draws the bars in a Matplotlib figure, updating the current figure."""
    ax.clear()  # Clear the current axes
    bar_colors = ['lightblue' for _ in range(len(arr))]
    if step < len(arr):  # Highlight only the current element
        bar_colors[step] = 'lightgreen'
    ax.bar(range(len(arr)), arr, color=bar_colors)
    ax.set_title("Sorting Visualization")
    ax.set_xlabel("Array Index")
    ax.set_ylabel("Array Value")
    ax.set_xlim(-1, len(arr))
    ax.set_ylim(0, 100)
    ax.grid(axis='y')

    # Update the same figure within Streamlit
    placeholder.pyplot(fig)

# Sorting Algorithms
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
            yield j

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        yield i

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        yield i

def merge_sort(arr, start=0, end=None):
    if end is None:
        end = len(arr) - 1
    if start < end:
        mid = (start + end) // 2
        yield from merge_sort(arr, start, mid)
        yield from merge_sort(arr, mid + 1, end)
        yield from merge(arr, start, mid, end)

def merge(arr, start, mid, end):
    left = arr[start:mid + 1]
    right = arr[mid + 1:end + 1]
    k = start
    while left and right:
        if left[0] <= right[0]:
            arr[k] = left.pop(0)
        else:
            arr[k] = right.pop(0)
        yield k
        k += 1
    while left:
        arr[k] = left.pop(0)
        yield k
        k += 1
    while right:
        arr[k] = right.pop(0)
        yield k
        k += 1

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1  # Only return the partition index

def quick_sort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low < high:
        pi = partition(arr, low, high)  # Get the partition index
        yield pi
        yield from quick_sort(arr, low, pi - 1)  # Sort left half
        yield from quick_sort(arr, pi + 1, high)  # Sort right half




def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(arr, n, i)
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        yield i
        yield from heapify(arr, i, 0)

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        yield i
        yield from heapify(arr, n, largest)

def counting_sort(arr):
    max_value = max(arr)
    count = [0] * (max_value + 1)
    output = [0] * len(arr)
    for i in range(len(arr)):
        count[arr[i]] += 1
    for i in range(1, max_value + 1):
        count[i] += count[i-1]
    for i in range(len(arr)-1, -1, -1):
        output[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1
        yield i
    for i in range(len(arr)):
        arr[i] = output[i]
        yield i

def radix_sort(arr):
    max_value = max(arr)
    exp = 1
    while max_value // exp > 0:
        yield from counting_sort_exp(arr, exp)
        exp *= 10

def counting_sort_exp(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    for i in range(1, 10):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        yield i
    for i in range(n):
        arr[i] = output[i]
        yield i

def bucket_sort(arr):
    max_value = max(arr)
    size = max_value // len(arr) + 1
    buckets = [[] for _ in range(size)]
    for i in arr:
        buckets[i // size].append(i)
    for bucket in buckets:
        bucket.sort()
    output = [num for bucket in buckets for num in bucket]
    for i in range(len(arr)):
        arr[i] = output[i]
        yield i

# Main Application
st.title("Sorting Algorithm Visualizer")
st.write("Select a sorting algorithm and speed to visualize the sorting process.")

# User inputs
algorithm = st.selectbox("Choose a sorting algorithm:", [
    "Bubble Sort", "Selection Sort", "Insertion Sort", "Merge Sort", 
    "Quick Sort", "Heap Sort", "Counting Sort", "Radix Sort", "Bucket Sort"
])
array_size = st.slider("Select the array size:", 5, 100, 20)
speed = st.selectbox("Select speed:", ["Slow", "Medium", "Fast"])

# Speed settings
if speed == "Slow":
    delay = 0.1
elif speed == "Medium":
    delay = 0.05
else:
    delay = 0.01
    
custom_array_input = st.text_input("Enter your custom array (comma-separated values):", "")
if st.button("Create Custom Array"):
    try:
        # Split the input by commas and attempt to convert to integers
        custom_array = [int(x.strip()) for x in custom_array_input.split(',')]
        
        # Check for negative integers
        if any(x < 0 for x in custom_array):
            st.error("Array must contain only non-negative integers.")
        else:
            st.session_state['array'] = custom_array
            st.success("Custom array created successfully!")

    except ValueError:
        st.error("Jahil. You have entered letters or special characters.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Generate random array
if st.button("Generate Array"):
    arr = np.random.randint(1, 101, size=array_size).tolist()
    st.session_state['array'] = arr

if 'array' not in st.session_state:
    st.session_state['array'] = []

# Visualize Sorting
placeholder = st.empty()  # Single display box placeholder
if st.button("Sort"):
    if st.session_state['array']:
        arr = st.session_state['array']
        progress_bar = st.progress(0)

        # Prepare for the sorting algorithm
        if algorithm == "Bubble Sort":
            sorter = bubble_sort(arr)
        elif algorithm == "Selection Sort":
            sorter = selection_sort(arr)
        elif algorithm == "Insertion Sort":
            sorter = insertion_sort(arr)
        elif algorithm == "Merge Sort":
            sorter = merge_sort(arr)
        elif algorithm == "Quick Sort":
            sorter = quick_sort(arr)
        elif algorithm == "Heap Sort":
            sorter = heap_sort(arr)
        elif algorithm == "Counting Sort":
            sorter = counting_sort(arr)
        elif algorithm == "Radix Sort":
            sorter = radix_sort(arr)
        elif algorithm == "Bucket Sort":
            sorter = bucket_sort(arr)

        # Sort the array and update progress
        steps = 0
        total_steps = len(arr) * (len(arr) - 1) // 2  
        for step in sorter:
            steps += 1
            progress = min(steps / total_steps, 1)  
            progress_bar.progress(progress)

            draw_bars(arr, step)
            time.sleep(delay)
        progress_bar.progress(1.0) 