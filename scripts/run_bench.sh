NTHREADS=1

export OMP_NUM_THREADS=$NTHREADS
export MKL_NUM_THREADS=$NTHREADS
export OPENBLAS_NUM_THREADS=$NTHREADS
export NUMEXPR_NUM_THREADS=$NTHREADS

python src/main.py \
  --matrix-kind structural \
  --orderings natural,amd,metis,nesdis \
  --nthreads $NTHREADS \
  --repeats 10 \
  --nmats 100
