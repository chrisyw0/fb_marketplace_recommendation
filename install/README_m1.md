## Instruction for installing Tensorflow and Pytorch

### Tensorflow
1. Install miniforge
2. Create and activate conda environment using command 

`conda env create -n <env name>`
`conda activate <env name>`

3. Install Python

`conda install python=3.9`

4. Install tensorflow 

`conda install -c apple tensorflow-deps`
`pip install tensorflow-macos`
`pip install tensorflow-metal`

5. Find the link for tensorflow-text matching your tensorflow and python version here
https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases

Install the wheel, e.g.

`pip install https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases/download/v2.9/tensorflow_text-2.9.0-cp39-cp39-macosx_11_0_arm64.whl`

6. Install tensorflow-hub and tf-models-official, as we install tensorflow-macos install of tensorflow, we should have --no-deps

`pip install --no-deps tensorflow-hub tf-models-official`

7. Install the remaining packages

`pip install -r requirement_tf.txt`


### Pytorch 

1. Setup conda (same steps as Tensorflow 1-3)
2. Pytorch 1.12 has MPS support so you can either choose stable version of nightly build version

`conda install pytorch torchvision -c pytorch`
`conda install pytorch torchvision -c pytorch-nightly`

3. Install the remaining packages

`pip install -r requirement_pt.txt`