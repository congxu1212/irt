# IRT-Router

## Experiments

### Training

We provide training data in the following file:

- **`data/train.csv`**: Training dataset.

You can train the **M-IRT router** using the following command:

```bash
python train_mirt.py
```

Similarly, to train the **N-IRT router**, run:

```bash
python train_nirt.py
```

We also provide a trained model checkpoint:
- **`mirt_bert.snapshot`**: uses bert-base-uncased as embedding model.

### Testing

- **`data/test1.csv`**: In-distribution test set.
- **`data/test2.csv`**: Out-of-distribution test set.

To evaluate the M-IRT router on the in-distribution test set, use the following command:

```bash
python test_router.py --router mirt --emb_name bert --test_path test1 --a 0.8 --lamda 0.3
```

Alternatively, you can execute the pre-written script:
```bash
sh test.sh
```





## Router Usage

To be continued…