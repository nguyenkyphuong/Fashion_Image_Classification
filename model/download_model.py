import os, subprocess, sys
FILE_ID = "https://drive.google.com/file/d/1sxWhzzHRVBv3RgFXpaNmrMTPnW3fgdA2/view?usp=sharing"
OUT = "fashion_mnist_model.keras"

if not os.path.exists(OUT):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    subprocess.check_call(["gdown", "--id", FILE_ID, "-O", OUT])
    print("Downloaded:", OUT)
else:
    print("Model already exists.")