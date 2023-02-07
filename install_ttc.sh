cd $OUT_DIR
git clone https://github.com/gaxler/triton.git
cd triton/python
git checkout aot-mlir-backend 
pip install cmake
pip install -e .
pip install regex
