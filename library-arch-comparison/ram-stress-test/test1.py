import sys
import numpy as np

def stress_memory():
    try:
        # Start consuming memory
        array_size = 10000
        data = np.random.random((array_size, array_size))
        count = 1
        while True:
            data = data * count
            np_size = data.itemsize*data.size/1000000000
            print(f"{count} The memory size of numpy array arr is: {np_size} GB")
            count+=1
    except MemoryError:
        print("Memory error occurred. RAM stress test completed.")

if __name__ == '__main__':
    print("Starting RAM stress test...")
    stress_memory()