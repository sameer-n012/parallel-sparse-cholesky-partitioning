python src/main.py \
  --matrix-kind curated_spd \
  --limit 5 \
  --orderings natural,amd,metis,nesdis \
  --threads 1,2,4,8,16 \
  --repeats 3 \
  --out-dir results
