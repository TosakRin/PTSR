conda create -n temp python=3.11 -y
pip install -r requirements.txt

# https://pytorch.org/get-started/previous-versions/
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
