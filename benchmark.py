# Imports
import pandas as pd
import numpy as np
import tracemalloc
import sys

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from time import perf_counter

# Setup
df = pd.read_csv("./data.csv")
catalog = df.groupby("product_id").first().reset_index()
products = catalog["product_id"].tolist()
cats = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(catalog[["category", "subcategory", "brand"]])
nums = MinMaxScaler().fit_transform(catalog[["final_price", "rating", "discount"]])
# Functions
# Example from the office hour
def get_features_base(x1=cats, x2=nums):
    features = list(x1.T) + list(x2.T)
    features = pd.DataFrame(features).T.values
    return features
# Pandas without transposing
def get_features_pandas(x1=cats, x2=nums):
    features = pd.concat(
        [pd.DataFrame(x1), pd.DataFrame(x2)], axis=1, ignore_index=True
    ).values
    return features
# Numpy stacking
def get_features_numpy(x1=cats, x2=nums):
    features = np.hstack([x1, x2])
    return features
# Simple, repeated timing and memory bechmark 
def benchmark(f, n):
    times = list()
    memory_peaks = []
    tracemalloc.start()
    
    for _ in range(n):
        tracemalloc.clear_traces()
        
        time_start = perf_counter()
        f()
        time_stop = perf_counter()
        
        _, peak = tracemalloc.get_traced_memory()
        
        times.append(time_stop - time_start)
        memory_peaks.append(peak)
    tracemalloc.stop()
    
    memory_peaks_mb = [mem / 10**6 for mem in memory_peaks]
    
    print(f"Average execution took:\t{np.mean(times):.4f}s (+/-{np.std(times):.4f}s)")
    print(f"Average peak memory:\t{np.mean(memory_peaks_mb):.4f} MB (+/-{np.std(memory_peaks_mb):.4f} MB)")
    
    return times, memory_peaks_mb
# Run benchmarks
if __name__ == '__main__':
    n_samples = 10
    
    if len(sys.argv) > 1:
        try:
            n_samples = abs(int(sys.argv[1])) # Catch negative numbers
        except:
            print("Invalid number of repetitions provided, defaulting to 10")
            
    to_benchmark = {
        'baseline': get_features_base,
        'pandas concat': get_features_pandas,
        'numpy vstack': get_features_numpy
    }
    for (name, func) in to_benchmark.items():
        print(f"Running benchmark (n={n_samples}): {name}")
        _ = benchmark(func, n_samples)