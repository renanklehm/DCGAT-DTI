import numpy as np
import pandas as pd
import sys
import ray
import hydra
import itertools
from datetime import datetime
import logging
import os 
import yaml
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
import random
from rdkit import DataStructs,Chem
from rdkit.Chem import rdFingerprintGenerator
random.seed()


def convert_y_unit(y, from_, to_):
	array_flag = False
	if isinstance(y, (int, float)):
		y = np.array([y])
		array_flag = True
	y = y.astype(float)    
	# basis as nM
	if from_ == 'nM':
		y = y
	elif from_ == 'p':
		y = 10**(-y) / 1e-9

	if to_ == 'p':
		zero_idxs = np.where(y == 0.)[0]
		y[zero_idxs] = 1e-10
		y = -np.log10(y*1e-9)
	elif to_ == 'nM':
		y = y
        
	if array_flag:
		return y[0]
	return y

    
#TODO: no need for split and new_split anymore
def get_dataset(cfg,X_drug, X_target, DTI,ddi=None,skipped=None):
    if cfg['datamodule']['splitting']['ratio']:
        print("splitting dataset using {} splitting strategy".format(cfg['datamodule']['splitting']['splitting_strategy']))
        dataset = split(cfg['datamodule']['splitting'],X_drug,X_target,DTI,ddi,skipped)
    else:
        print("Dataset already splitted. Skipping... ")
        dataset = pre_split(X_drug,X_target,DTI,ddi,skipped)
    return dataset

def new_balancing(y, ratio=1, rng=None):
    if rng is None:
        rng = np.random
    ind_1 = np.where(y.label==1)[0]
    ind_0 = np.where(y.label==0)[0]
    if len(ind_0) > ratio*len(ind_1):
        ind_0 = rng.choice(ind_0, ratio*len(ind_1), replace=False)
    ind = np.concatenate((ind_1,ind_0))
    rng.shuffle(ind)
    y = y.iloc[ind]
    return y  


def _cold_full_partition_entities(entity_ids, ratio, split_seed):
    train_val_ids, test_ids = train_test_split(
        entity_ids,
        test_size=ratio[2],
        random_state=split_seed,
    )
    val_seed = None if split_seed is None else split_seed + 1
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=ratio[1],
        random_state=val_seed,
    )

    labels = np.empty(len(entity_ids), dtype=np.int8)
    labels[np.searchsorted(entity_ids, train_ids)] = 0
    labels[np.searchsorted(entity_ids, val_ids)] = 1
    labels[np.searchsorted(entity_ids, test_ids)] = 2
    return labels


def _cold_full_split_scores(entity_ids, row_entity_ids, other_labels):
    scores = (
        pd.DataFrame({"entity": row_entity_ids, "other_split": other_labels})
        .groupby(["entity", "other_split"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=entity_ids, fill_value=0)
        .reindex(columns=[0, 1, 2], fill_value=0)
    )
    return scores.to_numpy(dtype=np.int64)


def _cold_full_improve_labels(scores, labels, rng, max_passes=6):
    labels = labels.copy()
    split_pairs = [(0, 1), (0, 2), (1, 2)]

    for _ in range(max_passes):
        improved = False
        pair_order = list(split_pairs)
        rng.shuffle(pair_order)

        for left_split, right_split in pair_order:
            left_idx = np.flatnonzero(labels == left_split)
            right_idx = np.flatnonzero(labels == right_split)
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue

            left_gain = scores[left_idx, right_split] - scores[left_idx, left_split]
            right_gain = scores[right_idx, left_split] - scores[right_idx, right_split]

            left_mask = left_gain > 0
            right_mask = right_gain > 0
            if not left_mask.any() or not right_mask.any():
                continue

            left_candidates = left_idx[left_mask]
            right_candidates = right_idx[right_mask]
            left_gain = left_gain[left_mask]
            right_gain = right_gain[right_mask]

            left_order = np.argsort(left_gain)[::-1]
            right_order = np.argsort(right_gain)[::-1]
            pair_count = min(len(left_order), len(right_order))

            for pair_idx in range(pair_count):
                if left_gain[left_order[pair_idx]] + right_gain[right_order[pair_idx]] <= 0:
                    break
                labels[left_candidates[left_order[pair_idx]]] = right_split
                labels[right_candidates[right_order[pair_idx]]] = left_split
                improved = True

        if not improved:
            break

    return labels


def _cold_full_materialize_splits(y, drug_ids, drug_labels, target_ids, target_labels):
    drug_lookup = pd.Series(drug_labels, index=drug_ids)
    target_lookup = pd.Series(target_labels, index=target_ids)
    row_drug_labels = y['Drug_ID'].map(drug_lookup).to_numpy(dtype=np.int8)
    row_target_labels = y['Prot_ID'].map(target_lookup).to_numpy(dtype=np.int8)
    row_labels = np.where(row_drug_labels == row_target_labels, row_drug_labels, -1)
    return (
        y[row_labels == 0],
        y[row_labels == 1],
        y[row_labels == 2],
        int((row_labels == -1).sum()),
    )


def _cold_full_has_both_classes(split_df):
    labels = set(split_df['label'].astype(int).unique().tolist())
    return 0 in labels and 1 in labels


def _optimized_cold_full_split(config, y):
    split_seed = config.get('seed')
    drug_ids = np.sort(np.unique(y.Drug_ID)).astype(int, copy=False)
    target_ids = np.sort(np.unique(y.Prot_ID)).astype(int, copy=False)

    best_candidate = None
    best_score = None
    best_balanced_candidate = None
    best_balanced_score = None

    for attempt in range(6):
        attempt_seed = None if split_seed is None else split_seed + attempt * 97
        attempt_rng = np.random.RandomState(attempt_seed) if attempt_seed is not None else np.random

        drug_labels = _cold_full_partition_entities(drug_ids, config['ratio'], attempt_seed)
        target_labels = _cold_full_partition_entities(target_ids, config['ratio'], attempt_seed)

        # Alternate drug/protein reassignment while keeping per-split entity counts fixed.
        for _ in range(6):
            prev_drug_labels = drug_labels.copy()
            prev_target_labels = target_labels.copy()

            target_lookup = pd.Series(target_labels, index=target_ids)
            target_row_labels = y['Prot_ID'].map(target_lookup).to_numpy(dtype=np.int8)
            drug_scores = _cold_full_split_scores(drug_ids, y['Drug_ID'].to_numpy(), target_row_labels)
            drug_labels = _cold_full_improve_labels(drug_scores, drug_labels, attempt_rng)

            drug_lookup = pd.Series(drug_labels, index=drug_ids)
            drug_row_labels = y['Drug_ID'].map(drug_lookup).to_numpy(dtype=np.int8)
            target_scores = _cold_full_split_scores(target_ids, y['Prot_ID'].to_numpy(), drug_row_labels)
            target_labels = _cold_full_improve_labels(target_scores, target_labels, attempt_rng)

            if np.array_equal(drug_labels, prev_drug_labels) and np.array_equal(target_labels, prev_target_labels):
                break

        train_ind, val_ind, test_ind, unused_rows = _cold_full_materialize_splits(
            y,
            drug_ids,
            drug_labels,
            target_ids,
            target_labels,
        )

        score = (len(train_ind) + len(val_ind) + len(test_ind), len(val_ind), len(train_ind), -unused_rows)
        candidate = (train_ind, val_ind, test_ind, unused_rows)
        if best_score is None or score > best_score:
            best_candidate = candidate
            best_score = score

        if all(len(split_df) > 0 for split_df in candidate[:3]) and (
            not config['balanced'] or all(_cold_full_has_both_classes(split_df) for split_df in candidate[:3])
        ):
            if best_balanced_score is None or score > best_balanced_score:
                best_balanced_candidate = candidate
                best_balanced_score = score

    if best_balanced_candidate is not None:
        return best_balanced_candidate
    if best_candidate is None:
        raise ValueError('cold_full could not construct any train/val/test split candidate.')
    return best_candidate

def split(config,X_drug,X_target,y,ddi,skipped):
    split_seed = config.get('seed')
    rng = np.random.RandomState(split_seed) if split_seed is not None else np.random

    #add rows and columns at skipped index with all 0
    if skipped:
        for i in skipped:
            ddi = np.insert(ddi,i,0,axis=0)
            ddi = np.insert(ddi,i,0,axis=1)
    

    '''
    ppi = pd.read_csv('/data/tanvir/DTI/comppi--interactions--tax_hsapiens_loc_all.txt', sep='\t',usecols=[0,4,8],on_bad_lines='skip')
    #ppi = ppi[ppi['Interaction Score'] >= 0.5]
    ppi = ppi[ppi['Protein A'].isin(X_target.index)]
    ppi = ppi[ppi['Protein B'].isin(X_target.index)]
    
    #check if any row has same value for both columns
    ppi = ppi[ppi['Protein A'] != ppi['Protein B']]
    
    #create pivot table of ppi
    ppi = ppi.pivot(index='Protein A',columns='Protein B',values='Interaction Score')
    #if a columns is absent in rows, add that column as an row
    for col in ppi.columns:
        if col not in ppi.index:
            ppi.loc[col] = ppi[col]
    
    #if a row is absent in columns, add that row as a column
    for row in ppi.index:
        if row not in ppi.columns:
            ppi[row] = ppi.loc[row]

    #if a row in X_target is not in ppi, add that to both row and column of ppi with all nan
    for row in X_target.index:
        if row not in ppi.index:
            ppi.loc[row] = np.nan
            ppi[row] = np.nan

    #sort index and columns
    ppi = ppi.sort_index(axis=0)
    ppi = ppi.sort_index(axis=1)

    #replace nan with 0
    ppi = ppi.fillna(0)
    X_target = X_target.sort_index(axis=0)

    #process drug-drug interaction
    ddi = pd.read_csv('/data/tanvir/DTI/ChCh-Miner_durgbank-chem-chem.tsv',sep='\t',on_bad_lines='skip')
    ddi['C'] = 1
    ddi = ddi[ddi['A'].isin(X_drug.index)]
    ddi = ddi[ddi['B'].isin(X_drug.index)]
    
    #check if any row has same value for both columns
    ddi = ddi[ddi['A'] != ddi['B']]
    #create pivot table of ppi
    ddi = ddi.pivot(index='A',columns='B',values='C')
    
    #if a columns is absent in rows, add that column as an row
    for col in ddi.columns:
        if col not in ddi.index:
            ddi.loc[col] = ddi[col]
    
    #if a row is absent in columns, add that row as a column
    for row in ddi.index:
        if row not in ddi.columns:
            ddi[row] = ddi.loc[row]

    #if a row in X_target is not in ppi, add that to both row and column of ppi with all nan
    for row in X_drug.index:
        if row not in ddi.index:
            ddi.loc[row] = np.nan
            ddi[row] = np.nan

    #sort index and columns
    ddi = ddi.sort_index(axis=0)
    ddi = ddi.sort_index(axis=1)

    #replace nan with 0
    ddi = ddi.fillna(0)
    X_drug = X_drug.sort_index(axis=0)

    '''
    X_drug['index'] = np.arange(len(X_drug))
    X_target['index'] = np.arange(len(X_target))
    y['Drug_ID'] = y['Drug_ID'].map(X_drug['index'])
    y['Prot_ID'] = y['Prot_ID'].map(X_target['index'])

    if config['splitting_strategy'] == 'random':

        train_val_ind, test_ind = train_test_split(
            np.arange(len(y)),
            test_size=config['ratio'][2],
            random_state=split_seed,
        )
        val_seed = None if split_seed is None else split_seed + 1
        train_ind, val_ind = train_test_split(
            train_val_ind,
            test_size=config['ratio'][1],
            random_state=val_seed,
        )
        train_ind = y.iloc[train_ind]
        val_ind = y.iloc[val_ind]
        test_ind = y.iloc[test_ind]      
    
    elif config['splitting_strategy'] == 'cold_drug':
        train_val_drug, test_drug = train_test_split(
            np.arange(len(np.unique(y.Drug_ID))),
            test_size=config['ratio'][2],
            random_state=split_seed,
        )
        val_seed = None if split_seed is None else split_seed + 1
        train_drug, val_drug = train_test_split(
            train_val_drug,
            test_size=config['ratio'][1],
            random_state=val_seed,
        )
        train_ind = y[y['Drug_ID'].isin(train_drug)]
        val_ind = y[y['Drug_ID'].isin(val_drug)]
        test_ind = y[y['Drug_ID'].isin(test_drug)]

    elif config['splitting_strategy'] == 'cold_target':
        train_val_target, test_target = train_test_split(
            np.arange(len(np.unique(y.Prot_ID))),
            test_size=config['ratio'][2],
            random_state=split_seed,
        )
        val_seed = None if split_seed is None else split_seed + 1
        train_target, val_target = train_test_split(
            train_val_target,
            test_size=config['ratio'][1],
            random_state=val_seed,
        )
        train_ind = y[y['Prot_ID'].isin(train_target)]
        val_ind = y[y['Prot_ID'].isin(val_target)]
        test_ind = y[y['Prot_ID'].isin(test_target)]

    elif config['splitting_strategy'] == 'cold_full':
        train_ind, val_ind, test_ind, unused_rows = _optimized_cold_full_split(config, y)
        if unused_rows:
            print(
                f'cold_full dropped {unused_rows} interaction rows that mixed '
                'different drug/protein holdout partitions.'
            )
        if len(train_ind) == 0 or len(val_ind) == 0 or len(test_ind) == 0:
            raise ValueError(
                'cold_full produced an empty train/val/test split. '
                'Try a different seed or larger validation/test ratios.'
            )

    if config['balanced']:
        train_ind = new_balancing(train_ind, rng=rng)
        val_ind = new_balancing(val_ind, rng=rng)
        test_ind = new_balancing(test_ind, rng=rng)
        if config['splitting_strategy'] == 'cold_full' and (len(train_ind) == 0 or len(val_ind) == 0 or len(test_ind) == 0):
            raise ValueError(
                'cold_full balancing removed all rows from at least one split. '
                'Try a different seed, larger validation/test ratios, or an unbalanced cold_full scenario.'
            )
    elif config['unbalanced_ratio']:
        train_ind = new_balancing(train_ind, config['unbalanced_ratio'], rng=rng)
        val_ind = new_balancing(val_ind, config['unbalanced_ratio'], rng=rng)
        test_ind = new_balancing(test_ind, config['unbalanced_ratio'], rng=rng)

    print(f'Number of samples in training: {len(train_ind)}')
    print(f'Number of samples in validation: {len(val_ind)}')
    print(f'Number of samples in test: {len(test_ind)}')

    X_drug = X_drug.drop(columns=['index'])
    X_target = X_target.drop(columns=['index'])
    dataset = {}
    dataset['train'] = train_ind
    dataset['val'] = val_ind
    dataset['test'] = test_ind
    dataset['X_drug'] = X_drug
    dataset['X_target'] = X_target
    #dataset['ppi'] = ppi
    dataset['ddi'] = ddi
    
    return dataset

  
def get_ddi(df):
    ms = [Chem.MolFromSmiles(x) for x in df.SMILES]
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    #fps = [fpgen.GetFingerprint(x) for x in ms]
    fps = []
    skipped = []
    for e, x in enumerate(ms):
        try:
            fps.append(fpgen.GetFingerprint(x))
        except:
            print('error, skipped...')
            skipped.append(e)
    #generate a similarity matrix of fps by fps
    similarity = np.array([[DataStructs.FingerprintSimilarity(i,j) for i in fps] for j in fps])
    #make the diagonal elements 0
    np.fill_diagonal(similarity,0)
    return similarity, skipped



def pre_split(X_drug,X_target,y,ddi,skipped):

    ind = np.arange(len(y))
    np.random.shuffle(ind)
    y = y.iloc[ind]

    X_drug['index'] = np.arange(len(X_drug))
    X_target['index'] = np.arange(len(X_target))
    y['Drug_ID'] = y['Drug_ID'].map(X_drug['index'])
    y['Prot_ID'] = y['Prot_ID'].map(X_target['index'])
    
    y = y.dropna()
    y['Drug_ID'] = y['Drug_ID'].astype(int)
    y['Prot_ID'] = y['Prot_ID'].astype(int)
    X_drug = X_drug.drop(columns=['index'])
    X_target = X_target.drop(columns=['index'])

    train_ind = y[y['indicator']==1]
    val_ind = y[y['indicator']==2]
    test_ind = y[y['indicator']==3]

    print(f'Number of samples in training: {len(train_ind)}')
    print(f'Number of samples in validation: {len(val_ind)}')
    print(f'Number of samples in test: {len(test_ind)}')

    dataset = {}
    dataset['train'] = train_ind
    dataset['val'] = val_ind
    dataset['test'] = test_ind
    dataset['X_drug'] = X_drug
    dataset['X_target'] = X_target
    dataset['ddi'] = ddi
    return dataset


# ############################################################


def setup_config_tune(config,cfg):
    if 'lr' in config.keys():
        cfg['module']['optimizer']['lr']: tune.loguniform = hydra.utils.instantiate(config['lr'])
    if 'batch_size' in config.keys():
        cfg['datamodule']['dm_cfg']['batch_size']: tune.choice = hydra.utils.instantiate(config['batch_size'])
    if 'layers' in config.keys():
        if 'categories' not in config['layers'].keys():
            random_combinations = []
            for L in range(config['layers']['min_hid_layers'], config['layers']['max_hid_layers']+1):
                for subset in itertools.permutations(config['layers']['layer_sizes'], L):
                    random_combinations.append(subset)
            # remove all the combinations that has previous element larger than the next element
            random_combinations = [i for i in random_combinations if all(i[j] >= i[j+1] for j in range(len(i)-1))]
            random_combinations = list(map(list, random_combinations))
            cfg['module']['network']['layers']: tune.choice = ray.tune.choice(random_combinations)
        else:
            cfg['module']['network']['layers']: tune.choice = hydra.utils.instantiate(config['layers'])
    if 'dropout' in config.keys():
        cfg['module']['network']['dropout']: tune.uniform = hydra.utils.instantiate(config['dropout'])
    if 'activation_fn' in config.keys():
        cfg['module']['network']['activation_fn']: tune.choice = hydra.utils.instantiate(config['activation_fn'])
    if 'weight_decay' in config.keys():
        cfg['module']['optimizer']['weight_decay']: tune.loguniform = hydra.utils.instantiate(config['weight_decay'])
    if 'optimizer' in config.keys():
        cfg['module']['optimizer']['optimizer']: tune.choice = hydra.utils.instantiate(config['optimizer'])
    if 'drug_gat' in config.keys():
        cfg['module']['GAT_params']['drug_gat']['out_channels']: tune.choice = hydra.utils.instantiate(config['drug_gat']['out_channels'])
        cfg['module']['GAT_params']['drug_gat']['heads']: tune.choice = hydra.utils.instantiate(config['drug_gat']['heads'])
        cfg['module']['GAT_params']['drug_gat']['dropout']: tune.choice = hydra.utils.instantiate(config['drug_gat']['dropout'])
        cfg['module']['GAT_params']['drug_gat']['add_self_loops']: tune.choice = hydra.utils.instantiate(config['drug_gat']['add_self_loops'])
        cfg['module']['GAT_params']['drug_gat']['num_layers']: tune.choice = hydra.utils.instantiate(config['drug_gat']['num_layers'])
    if 'prot_gat' in config.keys():
        cfg['module']['GAT_params']['prot_gat']['out_channels']: tune.choice = hydra.utils.instantiate(config['prot_gat']['out_channels'])
        cfg['module']['GAT_params']['prot_gat']['heads']: tune.choice = hydra.utils.instantiate(config['prot_gat']['heads'])
        cfg['module']['GAT_params']['prot_gat']['dropout']: tune.choice = hydra.utils.instantiate(config['prot_gat']['dropout'])
        cfg['module']['GAT_params']['prot_gat']['add_self_loops']: tune.choice = hydra.utils.instantiate(config['prot_gat']['add_self_loops'])
        cfg['module']['GAT_params']['prot_gat']['num_layers']: tune.choice = hydra.utils.instantiate(config['prot_gat']['num_layers'])
    if 'drug_threshold' in config.keys():
        cfg['module']['GAT_params']['drug_gat']['threshold']: tune.choice = hydra.utils.instantiate(config['drug_threshold'])
    if 'prot_threshold' in config.keys():
        cfg['module']['GAT_params']['prot_gat']['threshold']: tune.choice = hydra.utils.instantiate(config['prot_threshold'])
    if 'alpha' in config.keys():
        cfg['module']['GAT_params']['concat']['alpha']: tune.choice = hydra.utils.instantiate(config['alpha'])
    if 'concat' in config.keys():
        cfg['module']['GAT_params']['concat']: tune.choice = hydra.utils.instantiate(config['concat'])
    return cfg


def update_best_param(cfg):
    # load appropritate file of best params 
    if not os.path.exists(cfg['best_param_path']+cfg['best_param_name'] ):
        print('best param file not found')
        return cfg
    best_params = yaml.load(open(cfg['best_param_path']+cfg['best_param_name'] ), Loader=yaml.FullLoader)
    for key in best_params.keys():
        exec(f"{key} = {best_params[key]}")
    return cfg


def instantiate_callbacks(callbacks_cfg):

    callbacks = []

    for _, cb_conf in callbacks_cfg.items():
        callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def get_logger(user_config):  
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    if 'GAT' in user_config.keys():
        model = 'GAT'
    else:
        model = 'MLP'

    if user_config['logger']['name'] == 'drugbank':
        if user_config['datamodule']['splitting']['balanced']:
            logger_dir = f"./logs/drugbank/{user_config['datamodule']['splitting']['splitting_strategy']}_1_1/{model}/run_{datetime.now().strftime('%Y%m%d-%H%M%S')}/"
        else:
            logger_dir = f"./logs/drugbank/{user_config['datamodule']['splitting']['splitting_strategy']}_1_{user_config['datamodule']['splitting']['unbalanced_ratio']}/{model}/run_{datetime.now().strftime('%Y%m%d-%H%M%S')}/"
    else:
        data = user_config['preprocess']['data_path'].split('/')[1]
        logger_dir = f"./logs/{user_config['logger']['name']}/{data}/run_{datetime.now().strftime('%Y%m%d-%H%M%S')}/"
        
    #create dir if not exist 
    if not os.path.exists(logger_dir):
        os.makedirs(logger_dir)

    #save logger
    logger.addHandler(logging.FileHandler(logger_dir+'log.txt', 'a'))

    #save config at logger_dir
    with open(logger_dir+'config.yaml', 'w') as f:
        yaml.dump(user_config, f)

    return logger, logger_dir

