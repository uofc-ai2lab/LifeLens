# Visual De-Identification
## Setup
1. Create and activate a Python virtual environment

```sh
python3.11 -m venv venv311
# On Windows:
venv311\Scripts\activate
# On Mac/Linux:
source venv311/bin/activate
```

2. Upgrade pip

```sh
python3 -m pip install --upgrade pip
```

3. Install requirements

```sh
pip install -r requirements.txt
python3 -m pip install deface
```

---

## Notes
### Hardware Acceleration Using CUDA for Nvidia GPUs:
We can speed up neural network inference by enabling the optional [ONNX Runtime](https://microsoft.github.io/onnxruntime/) backend of `deface`. For optimal performance you should install it with appropriate [Execution Providers](https://onnxruntime.ai/docs/execution-providers) for your system. If you have multiple Execution Providers installed, ONNX Runtime will try to automatically use the fastest one available.

With a CUDA-capable GPU, you can enable GPU acceleration by installing the relevant packages:
```sh
python3 -m pip install onnx onnxruntime-gpu
```
If the `onnxruntime-gpu` package is found and a GPU is available, the face detection network is automatically offloaded to the GPU. This can significantly improve the overall processing speed.


---

## Usage
In the directory of the `defacer.py` file, specify the input path to an image and an output endpoint:
```sh
python3 defacer.py <input_file_path> -o <output_file_path>
```
For example:
```sh
python3 defacer.py test_data/input/city.jpg -o test_data/output/city_anon.jpg
``` 
- Note: The input is an existing image that hasn't yet been anonymized, but since the output is made using the model, a file name has to be specified in the `output_file_path`. Here "city_anon.jpg" was manually chosen as the output file name in the command line.
