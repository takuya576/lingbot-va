START=0
END=10

python -m evaluation.libero.client \
    --libero-benchmark libero_10 \
    --port 29056 \
    --test-num 50 \
    --task-range $START $END \
    --out-dir outputs/libero \
    --save-visualization True
