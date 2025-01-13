Creating the right Environment

python version: 3.10.16

conda install numpy=1.24 -c conda-forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install faiss-gpu -c conda-forge
pip install transformers
pip install librosa