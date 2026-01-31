import pickle
import os
from PIL import Image
import numpy as np

# Custom unpickler to handle NumPy compatibility
class NumpyCompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core.multiarray':
            module = 'numpy.core.multiarray'
        elif module == 'numpy._core.umath':
            module = 'numpy.core.umath'
        elif module == 'numpy.core':
            module = 'numpy.core'
        return super().find_class(module, name)

def display_aaaa_contents(sto_file_path):
    """
    Loads AAAA.sto and displays its current contents by index.
    """
    data = None
    try:
        print(f"Loading {os.path.basename(sto_file_path)} file...")
        print("=" * 60)
        with open(sto_file_path, 'rb') as f:
            unpickler = NumpyCompatibleUnpickler(f)
            data = unpickler.load()
        
        if data is None:
            print("Failed to load data from .sto file.")
            return

        print(f"Total items in {os.path.basename(sto_file_path)}: {len(data)}")
        print("=" * 60 + "\n")

        for i, item in enumerate(data):
            print(f"INDEX {i}:")
            print(f"Type: {type(item)}")
            print("-" * 40)
            if isinstance(item, np.ndarray):
                print(f"NumPy Array:")
                print(f"  Shape: {item.shape}")
                print(f"  Data type: {item.dtype}")
                print(f"  Size: {item.size} elements")
                print(f"  Memory: {item.nbytes} bytes")
                print(f"  Min value: {item.min()}")
                print(f"  Max value: {item.max()}")
                print(f"  Mean value: {item.mean():.2f}")
                print(f"  First 10 values: {item.flat[:10].tolist()}")
            elif isinstance(item, Image.Image):
                print(f"PIL Image:")
                print(f"  Mode: {item.mode}")
                print(f"  Size: {item.size}")
                print(f"  Format: {item.format}")
            elif isinstance(item, (list, tuple)):
                print(f"List with {len(item)} elements:")
                for j, sub_item in enumerate(item[:5]): # Display first 5 sub-items
                    print(f"    [{j}]: {type(sub_item)} - {sub_item}")
                if len(item) > 5:
                    print(f"    ...")
            else:
                print(f"Value: {item}")
                print(f"String representation: '{str(item)}'")
            print("\n")
        
        print("=" * 60)
        print(f"SUMMARY:")
        print(f"Total items: {len(data)}")
        print("\nQuick overview by index:")
        for i, item in enumerate(data):
            if isinstance(item, np.ndarray):
                print(f"  [{i}]: NumPy array {item.shape} ({item.dtype})")
            elif isinstance(item, Image.Image):
                print(f"  [{i}]: PIL Image {item.size} ({item.mode})")
            elif isinstance(item, (list, tuple)):
                print(f"  [{i}]: List with {len(item)} elements")
            else:
                print(f"  [{i}]: {type(item).__name__} - {str(item)[:20]}{'...' if len(str(item)) > 20 else ''}")
        print("=" * 60)

    except Exception as e:
        print(f"Error displaying file contents: {e}")

if __name__ == "__main__":
    sto_file = "/Users/pd3rvr/Documents/object_detection/multiwebapp/apps/material_detection_naturalobjects/spatiotemporal_model/sto_files/AAAA.sto"
    display_aaaa_contents(sto_file)
