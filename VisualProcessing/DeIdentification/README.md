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
In the directory of the `defacer.py` file, the input folder of images can be parsed with each image anonymized to an output folder:
```sh
python3 defacer.py <input_file_path> -r -O <output_directory_path>
```
For example:
```sh
python3 defacer.py test_data/input -r -O test_data/output
```
The output file name will be named `<basename>_anonymized<ext>` in this directory. For example, an input of `city.jpg` will output to `city_anonymized.jpg`.

Note: The `-r` tag is for recursive file processing, and the `-O` tag is to specify a directory to write anonymized outputs to.

Alternatively, you can specify a single image path for both the input and output:
```sh
python3 defacer.py <input_file_path> -o <output_file_path>
```
For example:
```sh
python3 defacer.py test_data/input/city.jpg -o test_data/output/city_anon.jpg
``` 
Note: The input is an existing image that hasn't yet been anonymized, but since the output is made using the model, a file name has to be specified in the `output_file_path`. Here "city_anon.jpg" was manually chosen as the output file name in the command line.

If you wish to test commands without any execution, there is a `dry-run` tag (`-n`) to list files that would be processed and exit (no changes). The tag should be included at the end of the command.

For example:
```sh
python3 defacer.py test_data/input/ -r -O test_data/output/ -n
```
