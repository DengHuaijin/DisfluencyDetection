#~/bin/bash

for dataset in train dev test train_sp; do
	sort -t " " -k 6n,6 "${dataset}.json" > tmp.json
	mv tmp.json "${dataset}.json"
done
