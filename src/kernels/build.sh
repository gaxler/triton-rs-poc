OUT=$OUT_DIR/$C_SOURCES_TARGET
TTC=$OUT_DIR/triton/python/triton/aot/ttc.py
TTL=$OUT_DIR/triton/python/triton/aot/ttl.py
TT_LINKER_PREFIX=$2

KERNEL_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

rm -rf $OUT
mkdir $OUT

$TT_PYTHON $TTC $KERNEL_PATH/vector_addition.py -n add_kernel --signature *fp32:16 *fp32:16 *fp32:16 i32:16 --BLOCK_SIZE 64 --out-name vec_add_64 -o $OUT/add_kernel64
$TT_PYTHON $TTC $KERNEL_PATH/vector_addition.py -n add_kernel --signature *fp32:16 *fp32:16 *fp32:16 i32:16 --BLOCK_SIZE 128 --out-name vec_add_128 -o $OUT/add_kernel128
$TT_PYTHON $TTL $(ls $OUT/*.h) -o $OUT/$1 --prefix $TT_LINKER_PREFIX 