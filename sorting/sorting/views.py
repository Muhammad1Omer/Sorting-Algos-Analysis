from django.shortcuts import render
from .models import RandomCollection, TimingsCollection
from django.http import JsonResponse


import random 
import numpy as np
from numba import njit, jit
import time
import psycopg2

# Create your views here
def homepage (request):
    return render(request, "home.html")

def empirical(request):
    return render(request, "empirical.html")

def Superset(request):
    return render(request, "Algo.html")

def Animations(request):
    return render(request, "interactive.html")

def theoretical(request):
    return render(request, "theoretical.html")


from django.shortcuts import redirect
import os
def AzeemView(request):
    streamlit_url = os.getenv("STREAMLIT_URL", "http://localhost:8501")
    return redirect(streamlit_url)


def dataset(size):
    data = [random.randint(-10**6, 10**6) for _ in range(size)]
    return sorted(data, reverse=True)



def GetData(id="00"):
    if id == "00":
        document = RandomCollection.find_one({"_id": id})
    elif id == "01":
        document = RandomCollection.find_one({"_id": id})
    
    return document




@jit(nopython=True)
def insertion(arr):
    n = len(arr)
    
    for i in range(n):
        key = arr[i]
        j = i - 1
        
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
            
        arr[j + 1] = key
        
    return arr

@jit(nopython=True)
def bubble(arr):
    
    for i in range (len(arr)):
        swaped = False
        for j in range (len(arr) - 1 ):
            if arr[j + 1] < arr[j]:
                temp = arr[j + 1]
                arr[j + 1] = arr[j]
                arr[j] = temp
                
                swaped = True
        
        i += 1
        
        if not swaped:
            break
        
                
    return arr

@jit(nopython=True)
def selection(arr):
    
    for i in range (len(arr)):
        MiniIdx = i
        for j in range (i, len(arr)):
            if arr[j] < arr[i]:
                MiniIdx = j
                
        temp = arr[i]
        arr[i] = arr[MiniIdx]
        arr[MiniIdx] = temp
        
    return arr

        
@jit(nopython=True)
def merge(arr):
    if len(arr) > 1:
        mid = len(arr) // 2 
        left = arr[:mid]    
        right = arr[mid:]
        
        merge(left)    
        merge(right)
    
        i = 0
        j = 0
        k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
            
        while i < len(left):
                arr[k] = left[i]
                i += 1
                k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1
        
    return arr



@njit
def counting(arr):
    max_val = max(arr)
    min_val = min(arr)
    range_of_elements = max_val - min_val + 1
    
    # Create count array and initialize it to 0
    count = [0] * range_of_elements
    output = [0] * len(arr)

    # Store the count of each element
    for num in arr:
        count[num - min_val] += 1

    # Modify the count array
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # Build the output array (avoiding reversed())
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i] - min_val] - 1] = arr[i]
        count[arr[i] - min_val] -= 1

    # Copy the sorted elements into original array
    for i in range(len(arr)):
        arr[i] = output[i]

    return arr

# ---------------- Quick sort ---------------------
@jit(nopython=True)
def median_of_three(arr, low, high):
    mid = (low + high) // 2
    first = arr[low]
    middle = arr[mid]
    last = arr[high]

    if (first <= middle <= last) or (last <= middle <= first):
        return mid 
    elif (middle <= first <= last) or (last <= first <= middle):
        return low  
    else:
        return high 

@jit(nopython=True)
def partition(arr, low, high):
    pivot_index = median_of_three(arr, low, high)
    pivot = arr[pivot_index]
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]  

    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  
    arr[i + 1], arr[high] = arr[high], arr[i + 1]  
    return i + 1

@jit(nopython=True)
def quick(arr, low, high):
    if low < high:
        pivot_index = partition(arr, low, high)
        quick(arr, low, pivot_index - 1)  
        quick(arr, pivot_index + 1, high)
        
    return arr


# ----------------- HEAP Sort -------------------

@jit(nopython=True)
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
        
        heapify(arr, n, largest)

@jit(nopython=True)
def heap(arr):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]  
        heapify(arr, i, 0)  

    return arr



# ---------------------- RADIX Sort -----------------------

@njit
def counting_sort_for_radix(arr, exp):
    n = len(arr)
    output = np.zeros(n, dtype=np.int64)
    count = np.zeros(10, dtype=np.int64)

    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    for i in range(n):
        arr[i] = output[i]

@njit
def radix_sort(arr):
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num

    exp = 1
    while max_val // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10

@njit
def radix(arr):
    n = len(arr)
    negatives = np.empty(n, dtype=np.int64)
    positives = np.empty(n, dtype=np.int64)
    neg_count = 0
    pos_count = 0

    for i in range(n):
        if arr[i] < 0:
            negatives[neg_count] = -arr[i]  # Convert negatives to positive
            neg_count += 1
        else:
            positives[pos_count] = arr[i]
            pos_count += 1

    negatives = negatives[:neg_count]
    positives = positives[:pos_count]

    # Apply radix sort to both
    if pos_count > 0:
        radix_sort(positives)
    if neg_count > 0:
        radix_sort(negatives)

    # Reverse the negatives array
    for i in range(neg_count // 2):
        negatives[i], negatives[neg_count - 1 - i] = negatives[neg_count - 1 - i], negatives[i]

    # Combine negatives and positives
    for i in range(neg_count):
        arr[i] = -negatives[i]  # Restore negatives
    for i in range(pos_count):
        arr[neg_count + i] = positives[i]
        
    return arr




# ------------------------ Bucket sort ----------------------------

@njit
def insertion_sort(arr, size):
    for i in range(1, size):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def bucket_sort(arr):
    size = len(arr)  
    buckets = [[] for _ in range(size)]  
    
    for i in arr:
        index = min(size - 1, int(i * size)) 
        buckets[index].append(i)

    # Sort each bucket individually using insertion sort
    for bucket in buckets:
        bucket.sort() 

    # Flatten the sorted buckets into a single output list
    output = [num for bucket in buckets for num in bucket]

    for i in range(len(arr)):
        arr[i] = output[i]

    return arr



def normalizeIt(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    normalized_arr = (arr - min_val) / (max_val - min_val)
    
    return normalized_arr


def TotalTimes(doc):
    
    times = []
    data = doc['datasets']  
    
    for arrSize in data:
        
        array = data[arrSize]

        start = time.time()
        insertion(array.copy()) 
        end = time.time()
        insertion_time = end - start

        start = time.time()
        bubble(array.copy())
        end = time.time()
        bubble_time = end - start

        start = time.time()
        selection(array.copy())
        end = time.time()
        selection_time = end - start
        
        
        
        start = time.time()
        merge(array.copy())
        end = time.time()
        merge_time = end - start
        
        start = time.time()
        quick(array.copy(), 0 , len(array) - 1)
        end = time.time()
        quick_time = end - start
        
        start = time.time()
        heap(array.copy())
        end = time.time()
        heap_time = end - start
        
        
        
        NormArr = normalizeIt(array.copy())
        
        start = time.time()
        bucket_sort(NormArr.copy())
        end = time.time()
        bucket_time = end - start
        
        start = time.time()
        counting(array.copy())
        end = time.time()
        count_time = end - start
        
        start = time.time()
        radix(array.copy())
        end = time.time()
        radix_time = end - start

        # Append the times to the list
        times.append({
            "size": arrSize,
            "insertion": insertion_time,
            "bubble": bubble_time,
            "selection": selection_time,
            "merge": merge_time,
            "quick": quick_time,
            "heap": heap_time,
            "bucket": bucket_time,
            "count": count_time,
            "radix": radix_time
        })

    return times



def calculate_times():
    document = GetData() 
    total_times = TotalTimes(document)

    TimingsCollection.update_one(
        {"_id": "00"},  
        {"$set": {"times": total_times}}, 
        upsert=True 
    )
    
    return True

def generate_dataset(id="00",SpecificSize = 0):
    if id == "00":
        sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
        datasets = {str(size): dataset(size) for size in sizes}
        
        # Save datasets to MongoDB
        RandomCollection.update_one(
            {"_id": "00" },  
            {"$set": {"datasets": datasets}},
            upsert=True
        )
        
        calculate_times() 
    
    elif id == "01":
        array = dataset(SpecificSize)
        RandomCollection.update_one(
            {"_id": "01"},
            {"$set": {"data": array}},
            upsert=True
        )


    
def get_timings(id="00"):
    timing = TimingsCollection.find_one({"_id": id})
    if timing:
        return timing  
    else:
        return {"error": "Timing not found"} 


    
def MonogoToPostgres(document_id = "00"):

    document = TimingsCollection.find_one({"_id": document_id})
    if document is None:
        print(f"No document found with _id: {document_id}")
        return 
    
    TimingsData = document['times'] 

    conn = psycopg2.connect(
    host="localhost",
    database="AlgoAss1",
    user="postgres",
    password="password"
    )

    cur = conn.cursor()
    for entry in TimingsData:
        size = entry['size']
        insertion_time = entry['insertion']
        bubble_time = entry['bubble']
        selection_time = entry['selection']
        
        merge_time = entry['merge']
        quick_time = entry['quick']
        heap_time = entry['heap']
        
        bucket_time = entry['bucket']
        count_time = entry['count']
        radix_time = entry['radix']
    
    
        cur.execute('''
        INSERT INTO timings (size, insertion, bubble, selection, merge, bucket, quick, heap, counting, radix)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (size) DO UPDATE SET
            insertion = EXCLUDED.insertion,
            bubble = EXCLUDED.bubble,
            selection = EXCLUDED.selection,
            merge = EXCLUDED.merge,
            bucket = EXCLUDED.bucket,
            quick = EXCLUDED.quick,
            heap = EXCLUDED.heap,
            counting = EXCLUDED.counting,
            radix = EXCLUDED.radix;
        ''', (size, insertion_time, bubble_time, selection_time, merge_time, bucket_time, quick_time, heap_time, count_time, radix_time))

    conn.commit()
    cur.close()
    conn.close()
