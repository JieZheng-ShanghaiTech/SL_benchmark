# Benchmarking of Machine Learning Methods for Predicting Synthetic Lethality Interactions

## Data Preparation and Download Instructions

Follow these steps to download and prepare the training data:

**Step 1**: Download all the data parts from the [Google Drive link](https://drive.google.com/drive/folders/1--hXtibIorfXt3jcKhz4btGH0YjqZafg?usp=sharing) provided in the repository.
(The actual command will depend on how you're downloading files from Google Drive)

```bash
# Step 2: Verify the integrity of the downloaded files.
md5sum -c md5sum.txt

# Step 3: Combine the parts into a single archive.
cat data_split* > data.tar.gz

# Step 4: Extract the dataset (the extracted folder will be approximately 90GB in size).
tar -xzvf data.tar.gz
```