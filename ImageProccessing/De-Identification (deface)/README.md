## Create and Activate Python Virtual Environment

```sh
python3.11 -m venv venv311
# On Windows:
venv311\Scripts\activate
# On Mac/Linux:
source venv311/bin/activate
```

## Upgrade pip

```sh
python.exe -m pip install --upgrade pip
```

## Install Dependencies

```sh
pip install -r requirements.txt
python3 -m pip install deface
```
## Hardware Acceleration Using CUDA for Nvidia GPUs
We can speed up neural network inference by enabling the optional [ONNX Runtime](https://microsoft.github.io/onnxruntime/) backend of `deface`. For optimal performance you should install it with appropriate [Execution Providers](https://onnxruntime.ai/docs/execution-providers) for your system. If you have multiple Execution Providers installed, ONNX Runtime will try to automatically use the fastest one available.

With a CUDA-capable GPU, you can enable GPU acceleration by installing the relevant packages:
```sh
python3 -m pip install onnx onnxruntime-gpu
```
If the `onnxruntime-gpu` package is found and a GPU is available, the face detection network is automatically offloaded to the GPU. This can significantly improve the overall processing speed.

## Usage
```sh
python3 defacer.py <input_file_path> -o <output-file_path>
```
For example:
```sh
python3 defacer.py test_data/input/two_peopple.jpg -o test_data/output/two_people_anon.jpg
``` 