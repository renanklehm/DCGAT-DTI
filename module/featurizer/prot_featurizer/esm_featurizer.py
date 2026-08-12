# import esm, torch, sys, os
# import numpy as np
# from tqdm import tqdm


# class ESMFEATURE:
#     def __init__(self,device):
#         self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#         self.model = self.model.to(device)
#         self.batch_converter = self.alphabet.get_batch_converter()
#         self.device = device

#     def get_representations(self, X_target):

#         data = []
#         for i in range(len(X_target)):
#             data.append(("protein"+str(i),X_target[i]))
        
#         batch_size = 1
#         data = [data[i * batch_size:(i + 1) * batch_size] for i in range((len(data) + batch_size - 1) // batch_size )]
#         # Process batches (this supports multiple sequence inputs)
#         self.model.eval()  # disables dropout for deterministic results

#         sequence_representations = []    
#         for temp_data in tqdm(data):
#             batch_labels, batch_strs, batch_tokens = self.batch_converter(temp_data)
#             batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
            
#             batch_tokens = batch_tokens.to(self.device)

#             # Extract per-residue representations (on CPU)
#             with torch.no_grad():
#                 results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
#             token_representations = results["representations"][33].to('cpu')
            
#             # Generate per-sequence representations via averaging
#             # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
#             for i, tokens_len in enumerate(batch_lens):
#                 sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
#                 # print sequence representation shape
                
#             del results, batch_tokens
#             torch.cuda.empty_cache() 

#         #use torch stack to convert list of tensors to tensor
#         sequence_representations = torch.stack(sequence_representations)

#         return np.array(sequence_representations)


import esm, torch
import numpy as np
from tqdm import tqdm


class ESMFEATURE:
    # ESM-2 uses rotary embeddings, so there is no length cap -- the only ceiling
    # is memory. Attention allocates B * heads * L^2 activations, so a fixed
    # batch size that is fine for short targets OOMs on long ones. Batches are
    # instead built under a budget on B * L^2.
    ATTENTION_BUDGET = 2e7

    def __init__(self, device, batch_size=4, attention_budget=None):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = self.model.to(device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = device
        self.batch_size = batch_size
        self.attention_budget = attention_budget or self.ATTENTION_BUDGET

    def _make_batches(self, order, lengths):
        """Group indices (pre-sorted by length) into length-homogeneous batches."""
        batches, current, current_max = [], [], 0
        for idx in order:
            candidate_max = max(current_max, lengths[idx])
            if current and (
                len(current) >= self.batch_size
                or (len(current) + 1) * candidate_max ** 2 > self.attention_budget
            ):
                batches.append(current)
                current, current_max = [idx], lengths[idx]
            else:
                current.append(idx)
                current_max = candidate_max
        if current:
            batches.append(current)
        return batches

    def _embed(self, indices, X_target):
        temp_data = [("protein" + str(i), X_target[i]) for i in indices]
        _, _, batch_tokens = self.batch_converter(temp_data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(self.device)

        with torch.inference_mode():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        out = {}
        for i, tokens_len in enumerate(batch_lens):
            out[indices[i]] = token_representations[i, 1 : tokens_len - 1].mean(0).cpu()

        del results, token_representations, batch_tokens
        return out

    def get_representations(self, X_target):

        lengths = [len(seq) for seq in X_target]
        # Sorting by length keeps padding (and therefore wasted attention) minimal.
        order = sorted(range(len(X_target)), key=lambda i: lengths[i])
        batches = self._make_batches(order, lengths)

        representations = {}
        for indices in tqdm(batches):
            pending = [indices]
            while pending:
                chunk = pending.pop()
                try:
                    representations.update(self._embed(chunk, X_target))
                except torch.cuda.OutOfMemoryError:
                    if len(chunk) == 1:
                        raise
                    # Budget was still too optimistic for this length; split and retry.
                    torch.cuda.empty_cache()
                    mid = len(chunk) // 2
                    pending.extend([chunk[:mid], chunk[mid:]])

        #use torch stack to convert list of tensors to tensor
        sequence_representations = torch.stack([representations[i] for i in range(len(X_target))])

        return np.array(sequence_representations)


