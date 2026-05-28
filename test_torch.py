import os
import sys

print("Python version:", sys.version)
print("Current Working Directory:", os.getcwd())

# 1. Let's find where PyTorch is installed and get its lib directory
try:
    import site
    site_packages = site.getsitepackages()
    print("Site packages directories:", site_packages)
except Exception as e:
    print("Could not get site packages:", e)
    site_packages = []

torch_lib = None
for sp in site_packages:
    candidate = os.path.join(sp, "torch", "lib")
    if os.path.exists(candidate):
        torch_lib = candidate
        break

if torch_lib:
    print(f"[*] Found torch lib directory: {torch_lib}")
    # Add it explicitly to DLL directory search paths
    os.add_dll_directory(torch_lib)
    print("[*] Successfully added torch/lib to DLL directories.")
else:
    print("[!] Warning: torch/lib directory not found in site-packages!")

# 2. Try importing torch
try:
    import torch
    print("[+] SUCCESS! PyTorch imported successfully!")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("[-] FAILED to import torch:")
    import traceback
    traceback.print_exc()
