NTHREADS=4

export OMP_NUM_THREADS=$NTHREADS
export MKL_NUM_THREADS=$NTHREADS
export OPENBLAS_NUM_THREADS=$NTHREADS
export NUMEXPR_NUM_THREADS=$NTHREADS

# python src/main.py \
#   --orderings natural,amd,metis,nesdis \
#   --nthreads $NTHREADS \
#   --repeats 10 \
#   --max-size 100000 \
#   --min-size 0 \
#   --max-nnz 1000000 \
#   --min-nnz 0 \
#   --nmats 200

python src/main_random.py \
  --orderings natural,amd,metis,nesdis \
  --nthreads $NTHREADS \
  --repeats 10 \
  --max-size 100000 \
  --min-size 10000 \
  --max-density 0.000001 \
  --min-density 0.00001 \
  --nmats 10
