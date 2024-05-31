# Benchmarking of Machine Learning Methods for Predicting Synthetic Lethality Interactions

## Data Preparation and Download Instructions

Follow these steps to download and prepare the training data:

**Step 1:** Download all the data parts from the [Google Drive link](https://drive.google.com/drive/folders/1--hXtibIorfXt3jcKhz4btGH0YjqZafg?usp=sharing) provided in the repository.

> \[!TIP]
>
> The actual command will depend on how you're downloading files from Google Drive.

**Step 2:** Combine the parts into a single archive.
```bash
cat data_large.tar.gz.part* > data_large.tar.gz     # Complete version
# cat data_small.tar.gz.part* > data_small.tar.gz   # The version without PiLSL database
```
**Step 3:** Verify the integrity of the downloaded files.
```bash
md5sum -c data_large.tar.gz.md5
# md5sum -c data_small.tar.gz.md5 # The version without PiLSL database
```
**Step 4:** Extract the dataset.
```bash
tar -xzvf data_large.tar.gz
# tar -xzvf data_small.tar.gz # The version without PiLSL database
```

> \[!IMPORTANT]
>
> The extracted folder will be approximately 90GB in size.
> Please ensure there is sufficient disk space.
