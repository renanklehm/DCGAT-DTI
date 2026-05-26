import importlib.util
import sys
import types

import numpy as np
import torch
from tqdm import tqdm


def _patch_incompatible_torchao() -> None:
    """Shadow torchao when it is installed against an incompatible torch build.

    Recent transformers releases import `torchao` from `modeling_utils`. On some
    environments `torchao` is present, but the matching torch build does not ship
    `torch.sparse._triton_ops_meta`, which makes any model import fail before the
    featurizer can load ChemBERT.
    """

    if "torchao" in sys.modules:
        return

    if importlib.util.find_spec("torchao") is None:
        return

    if importlib.util.find_spec("torch.sparse._triton_ops_meta") is not None:
        return

    quantization_module = types.ModuleType("torchao.quantization")

    class Int4WeightOnlyConfig:  # pragma: no cover - compatibility shim
        pass

    quantization_module.Int4WeightOnlyConfig = Int4WeightOnlyConfig

    torchao_module = types.ModuleType("torchao")
    torchao_module.quantization = quantization_module

    sys.modules["torchao"] = torchao_module
    sys.modules["torchao.quantization"] = quantization_module

class CHEMFEATURE:
    def __init__(self,device):
        _patch_incompatible_torchao()
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        #total parameters in the model
        #print(sum(p.numel() for p in self.model.parameters()))
        #sys.exit()
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        self.device = device

    def get_representations(self, X_drug):

        #print("drug is : ", X_drug)
    
        batch_size = 1
        data = [X_drug[i * batch_size:(i + 1) * batch_size] for i in range((len(X_drug) + batch_size - 1) // batch_size )]

        drug_representations = []
        for temp_data in tqdm(data):
            inputs = self.tokenizer(temp_data.tolist(), padding=True, truncation=True, return_tensors="pt").to(self.device)
            batch_lens = (inputs['attention_mask'] != 0).sum(1)
            
            #return hidden representations
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            token_representations = outputs.hidden_states[-1].to('cpu')

                #token_representations = model(**inputs).logits

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, tokens_len in enumerate(batch_lens):
                drug_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
            
            del token_representations, inputs
            torch.cuda.empty_cache()

        
      #  print((drug_representations))
        drug_representations = torch.stack(drug_representations)
        return np.array(drug_representations)

