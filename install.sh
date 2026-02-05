
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge -y

pip install -e .

# install pi and libero package
pip install -e ".[pi]"

# Fixed cmake by uninstalling pip version and installing conda version
pip uninstall cmake -y

git clone https://github.com/StanfordVL/egl_probe.git
cd egl_probe
# Edit CMakeLists.txt to use cmake_minimum_required(VERSION 3.5...3.29)
pip install -e .

cd ../ #go to lerobot/
pip install -e ".[libero]"

# For pi 0.5
pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"