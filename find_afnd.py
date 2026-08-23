import os
import sys

def find_file(filename, search_path):
    for root, dir, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

path = find_file("AFND_credible.json", "/home/assem-elqersh/Desktop")
if path:
    print(f"Found at {path}")
else:
    print("Not found")
