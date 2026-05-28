from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm

class CHEMFEATURE:
    def __init__(self, device, batch_size=32):
        self.model = AutoModelForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        #total parameters in the model
        #print(sum(p.numel() for p in self.model.parameters()))
        #sys.exit()
        self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        self.device = device
        self.batch_size = batch_size

    def get_representations(self, X_drug):

        #print("drug is : ", X_drug)

        batch_size = self.batch_size
        data = [X_drug[i * batch_size:(i + 1) * batch_size] for i in range((len(X_drug) + batch_size - 1) // batch_size)]

        drug_representations = []
        for temp_data in tqdm(data):
            inputs = self.tokenizer(temp_data.tolist(), padding=True, truncation=True, return_tensors="pt").to(self.device)
            batch_lens = (inputs['attention_mask'] != 0).sum(1)

            #return hidden representations
            with torch.inference_mode():
                outputs = self.model(**inputs, output_hidden_states=True)
            token_representations = outputs.hidden_states[-1]

            # Generate per-sequence representations via averaging
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            for i, tokens_len in enumerate(batch_lens):
                drug_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0).cpu())

            del outputs, token_representations, inputs

        #  print((drug_representations))
        drug_representations = torch.stack(drug_representations)
        return np.array(drug_representations)

