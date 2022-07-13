# Convert dataset to Pepper compatible format

Motivations:

- Isolate dataset conversion from the `Dataset` class.
- Previously (torch-reid), converts the dataset for training / testing when the dataset is called. However, I thought having a seaparate script would be beneficial (no need to wait, no need to pass arguments during training, etc...).
