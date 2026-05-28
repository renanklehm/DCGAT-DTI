# DCGAT-DTI: Dynamic Cross-Graph Attention for Drug–Target Interaction Prediction

**DCGAT-DTI** is a novel deep learning framework for **Drug–Target Interaction (DTI) prediction**, designed to enhance drug discovery by dynamically modeling interactions between chemical compounds and proteins. Unlike traditional methods that process drug and protein similarity graphs independently, **DCGAT-DTI** leverages a **Dynamic Cross-Graph Attention (DCGAT)** module to capture intra- and cross-graph dependencies.

## Key Features and Novelty
- **DCGAT Module**: Enables **cross-modal message passing** between drugs and proteins, allowing embeddings to dynamically **incorporate information across both modalities** through intra- and cross-graph attention mechanisms.
- **CNS Network (Cross Neighborhood Selection)**: A **GCN-based selection mechanism** that uses **Gumbel-Softmax Estimator** to  **dynamically selects cross-modal neighbors**, ensuring that each drug and protein node interacts with the most relevant counterparts.

## Complete Pipeline
![Complete Pipeline](DCGAT-DTI_(updated).svg)


## 📂 Dataset
The datasets used in this project can be downloaded from the following link:

👉 [Download Datasets](https://drive.google.com/file/d/1VzO6BQNEbbudYBeLoFG9fe5IrzMgflgn/view?usp=sharing)

Create a directory named **Datasets** by running the command :
```
mkdir Datasets
```

After downloading, extract the zip file in the **Datasets** directory. There are four datasets- BindingDB, DrugBank, Yamanishi_08 and Luo's.

---

## 🔧 Running Different Configurations
You can run the model in built-in warm, cold-drug, cold-target, and cold-full settings depending on whether the data is **balanced** or **unbalanced**.

## Common Entrypoint

You can now access the repo workflows through a single routed CLI:

```bash
python main.py --help
```

Available commands:

```bash
python main.py reproduce-paper --help
python main.py train-existing-eval-custom --help
python main.py train-custom --help
python main.py train-custom-test-custom --help
```

---

### 🧬 DrugBank Dataset
```bash
# Balanced - Warm Start
python run.py --config-name drugbank_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=random"

# Balanced - Cold Start for Drug
python run.py --config-name drugbank_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=cold_drug"

# Balanced - Cold Start for Protein
python run.py --config-name drugbank_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=cold_target"

# Balanced - Cold Start for Drug and Protein
python run.py --config-name drugbank_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=cold_full"

# Unbalanced - Warm Start
python run.py --config-name drugbank_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=random"

# Unbalanced - Cold Start for Drug
python run.py --config-name drugbank_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=cold_drug"

# Unbalanced - Cold Start for Protein
python run.py --config-name drugbank_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=cold_target"

# Unbalanced - Cold Start for Drug and Protein
python run.py --config-name drugbank_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=cold_full"
```

### 🧪 BindingDB Dataset
```bash
# Balanced - Warm Start
python run.py --config-name bindingDB_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=random"

# Balanced - Cold Start for Drug
python run.py --config-name bindingDB_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=cold_drug"

# Balanced - Cold Start for Protein
python run.py --config-name bindingDB_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=cold_target"

# Balanced - Cold Start for Drug and Protein
python run.py --config-name bindingDB_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=cold_full"

# Unbalanced - Warm Start
python run.py --config-name bindingDB_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=random"

# Unbalanced - Cold Start for Drug
python run.py --config-name bindingDB_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=cold_drug"

# Unbalanced - Cold Start for Protein
python run.py --config-name bindingDB_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=cold_target"

# Unbalanced - Cold Start for Drug and Protein
python run.py --config-name bindingDB_train_GAT.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=cold_full"
```

### 🧬 Yamanishi_08 Dataset
```bash
# Balanced - Warm Start
python run.py --config-name yamanishi_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=random"

# Balanced - Cold Start for Drug
python run.py --config-name yamanishi_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=cold_drug"

# Balanced - Cold Start for Protein
python run.py --config-name yamanishi_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=cold_target"

# Unbalanced - Warm Start
python run.py --config-name yamanishi_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=random"

# Unbalanced - Cold Start for Drug
python run.py --config-name yamanishi_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=cold_drug"

# Unbalanced - Cold Start for Protein
python run.py --config-name yamanishi_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=cold_target"
```

### 🧪 Luo's Dataset
```bash
# Balanced - Warm Start
python run.py --config-name luo_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=random"

# Balanced - Cold Start for Drug
python run.py --config-name luo_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=cold_drug"

# Balanced - Cold Start for Protein
python run.py --config-name luo_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=True" "datamodule.splitting.splitting_strategy=cold_target"

# Unbalanced - Warm Start
python run.py --config-name luo_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=random"

# Unbalanced - Cold Start for Drug
python run.py --config-name luo_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=cold_drug"

# Unbalanced - Cold Start for Protein
python run.py --config-name luo_train.yaml "tuning.param_search.tune=False" "datamodule.splitting.balanced=False" "datamodule.splitting.splitting_strategy=cold_target"
```

---


## Different Dataset Usage
Follow these steps to integrate your own custom dataset:

1. **Modify the Preprocessing Pipeline**  
   - Update `utils.PREPROCESS` to add your dataset name.

2. **Add a Preprocessing Script**  
   - The script should return the following data structures:

   ```python
   X_drug: nx1 pd.DataFrame  # Drug data
   # Index: Drug names
   # Column 1: SMILES sequence

   X_target: mx1 pd.DataFrame  # Protein data
   # Index: Target names
   # Column 1: Protein sequence

   DTI: mxn (available) pd.DataFrame  # Drug-Target Interaction
   # Index: 0-mxn
   # Column 1: Drug names matching X_drug index
   # Column 2: Target names matching X_target index
   # Column 3: Interaction label (0,1)
   ```

3. **Update Configuration**  
    - Add a new `train.yaml` (e.g., `bindingDB_train_GAT.yaml`) in the `configs/` directory to define preprocessing and datamodule settings.

## Quick Tutorial

The custom-data scripts expect a dataset with three fields in this order:

```text
smiles|sequence|activation
```

Example row:

```text
CCO|MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAY|1
```

You can also use JSON or Parquet records with `smiles`, `sequence`, and `activation` columns.

### 1. Train on an existing paper scenario and evaluate on your custom dataset

This keeps the original behavior of `train_existing_scenario_and_eval_custom.py`:

```bash
python main.py train-existing-eval-custom \
  --scenario drugbank:balanced_warm \
  --custom-data path/to/custom.csv \
  --has-header
```

You can also use the new built-in `cold_full` scenarios here:

```bash
python main.py train-existing-eval-custom \
  --scenario drugbank:balanced_cold_full \
  --custom-data path/to/custom.csv \
  --has-header
```

Outputs are written under `artifacts/custom_eval/...` and include:

- prepared custom tables
- exclusion reports for invalid rows
- prediction exports with probabilities
- the selected checkpoint
- a `.safetensors` export
- `metrics.json`

### 2. Train, evaluate, and export directly on your custom dataset

This is the new workflow added in this change. It trains on a split of your custom dataset, evaluates on the held-out test split, and exports predictions plus weights.

```bash
python main.py train-custom \
  --custom-data path/to/custom.csv \
  --has-header \
  --base-config drugbank_train_GAT.yaml \
  --split-strategy warm \
  --seed 42
```

When `--best-param-name` is omitted, custom-training workflows now use `configs/best_params/custom_default.yaml`.

Useful variants:

```bash
python main.py train-custom --custom-data path/to/custom.csv --has-header --split-strategy cold_drug
python main.py train-custom --custom-data path/to/custom.csv --has-header --split-strategy cold_target
python main.py train-custom --custom-data path/to/custom.csv --has-header --split-strategy cold_full
python main.py train-custom --custom-data path/to/custom.csv --has-header --no-balanced --unbalanced-ratio 10
python main.py train-custom --custom-data path/to/custom.csv --has-header --reuse-custom-embeddings
python main.py train-custom --custom-data path/to/custom.parquet --split-strategy warm
```

Outputs are written under `artifacts/custom_training/...` and include:

- `prepared_data/drug_table.tsv`
- `prepared_data/protein_table.tsv`
- `prepared_data/relation_table.tsv`
- `prepared_data/split_table.tsv`
- `logs/train.log`
- `exports/predictions_with_scores.csv`
- `exports/*.safetensors`
- `metrics.json`

The exported predictions file keeps the same custom-data-oriented format as the existing custom evaluation script, and in the new training flow it also adds a `split` column so you can see whether each kept row was used for `train`, `val`, or `test`.

`cold_full` is stricter than `cold_drug` and `cold_target`: it holds out both a set of drugs and a set of proteins from training. Rows that mix different holdout partitions are marked as `unused` in the predictions export and are not used for train/val/test metrics.

### 3. Train on one custom dataset and test on another custom dataset

This new workflow trains with the same built-in balancing and split modes, but evaluates on a separate custom file.

```bash
python main.py train-custom-test-custom \
  --train-data path/to/train_custom.csv \
  --train-has-header \
  --test-data path/to/test_custom.parquet \
  --base-config drugbank_train_GAT.yaml \
  --split-strategy warm \
  --seed 42
```

Useful variants:

```bash
python main.py train-custom-test-custom --train-data path/to/train.csv --train-has-header --test-data path/to/test.csv --test-has-header --split-strategy cold_drug
python main.py train-custom-test-custom --train-data path/to/train.csv --train-has-header --test-data path/to/test.csv --test-has-header --split-strategy cold_target
python main.py train-custom-test-custom --train-data path/to/train.csv --train-has-header --test-data path/to/test.csv --test-has-header --split-strategy cold_full
python main.py train-custom-test-custom --train-data path/to/train.csv --train-has-header --test-data path/to/test.json --no-balanced --unbalanced-ratio 10
```

Outputs are written under `artifacts/custom_cross_dataset/...` and include:

- prepared train and test tables
- train split table for the training dataset
- exclusion reports for invalid rows in both datasets
- prediction exports for the external test dataset
- the selected checkpoint
- a `.safetensors` export
- `metrics.json`

### 4. Reproduce the paper runs

```bash
python main.py reproduce-paper --datasets drugbank bindingDB
```

`reproduce-paper` now also includes the built-in `balanced_cold_full` and `unbalanced_cold_full` scenarios for DrugBank and BindingDB. Those runs currently reuse the closest existing cold-start best-parameter files because the repo does not yet have dedicated tuned `cold_full` parameter YAMLs.

---

## Integration of a Custom Featurizer
If you want to modify the drug or protein featurization, follow these steps:

1. **Add New Featurizers**  
   - Implement new featurization methods in `module.featurizer.drug_featurizer` and `module.featurizer.prot_featurizer`.  
   - Ensure the featurizers return:

     ```python
     nxq  # Drug embeddings (q = embedding size)
     mxp  # Protein embeddings (p = embedding size)
     ```

2. **Modify Configuration Files**  
   - Update `configs.featurizer` to reflect the new featurizers.
   - Set `drug_dim` and `prot_dim` in `configs.module.GAT`.

---

## Using a Custom Classifier
To integrate a new classification model:

1. **Create a New Model Pipeline**  
   - Add a new model file in the `module` directory (similar to `GAT.py` or `MLP.py`).

2. **Update Configuration Files**  
   - Add necessary settings in `configs.module`, following the structure of `config.module.GAT` for `GAT.py`.

---

### 📌 Notes
- Ensure that all custom implementations are compatible with the existing pipeline.
- Modify necessary configurations to properly register new data, featurizers, or classifiers.






