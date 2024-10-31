conda create -n PTSR python=3.11 -y
conda activate PTSR
pip install -r requirements.txt

# https://pytorch.org/get-started/previous-versions/
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
