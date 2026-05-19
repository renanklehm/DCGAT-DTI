import torch.nn as nn
import torch,hydra,sys
import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchmetrics 
from torch_geometric.nn import MLP,GCNConv
import torch
import torch.nn.functional as F
import torch
from typing import Optional
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn
import typing
from typing import Optional, Tuple, Union
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


torch.set_float32_matmul_precision('medium')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def contrastive_loss(z1, z2, maskbool, temperature=0.5):

    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)
    logits = torch.matmul(z1, z2.T) / temperature  
    exp_logits = torch.exp(logits) * maskbool 
    positive_loss = -torch.log(exp_logits.sum(dim=-1) + 1e-10)  
    negative_loss = torch.log(torch.exp(logits).sum(dim=-1) + 1e-10)
    loss = positive_loss + negative_loss
    return loss.mean()


def update_maskbool(maskbool, inv_drug, inv_target, y):

    updated_maskbool = maskbool.clone()
    count = 0 
    for drug_idx, prot_idx, label in zip(inv_drug, inv_target, y):

        if label == 1: 
            if updated_maskbool[drug_idx, prot_idx] == 1:
                count += 1 

            updated_maskbool[drug_idx, prot_idx] = 1  

    return updated_maskbool




class GATConv(MessagePassing):
    def __init__(
        self,
        in_channels_first: int,
        in_channels_second: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        self.lin_first_graph = Linear(in_channels_first, heads * out_channels, bias=False, weight_initializer='glorot')
        self.lin_second_graph = Linear(in_channels_second, heads * out_channels, bias=False, weight_initializer='glorot')

        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))
        self.att_second_graph = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False, weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_first_graph.reset_parameters()
        self.lin_second_graph.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_second_graph)
        zeros(self.bias)

    @overload
    def forward(
        self,
        x_first: Tensor,  # Node features for the first graph
        edge_index: Adj,
        x_second: Tensor,  # Node features for the second graph
        edge_attr: OptTensor = None,
        size: Size = None,
        mask: Optional[Tensor] = None,  
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x_first: Tensor,  # Node features for the first graph
        edge_index: Tensor,
        x_second: Tensor,  # Node features for the second graph
        edge_attr: OptTensor = None,
        size: Size = None,
        mask: Optional[Tensor] = None, 
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x_first: Tensor,  # Node features for the first graph
        edge_index: SparseTensor,
        x_second: Tensor,  # Node features for the second graph
        edge_attr: OptTensor = None,
        size: Size = None,
        mask: Optional[Tensor] = None,  
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass


    def forward(  # noqa: F811
        self,
        x_first: Tensor,  # Node features for the first graph
        edge_index: Adj,
        x_second: Tensor,  # Node features for the second graph
        edge_attr: OptTensor = None,
        size: Size = None,
        maskbool: Optional[Tensor] = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        H, C = self.heads, self.out_channels

        # Transform the node features for the first graph
        x_src = self.lin_first_graph(x_first).view(-1, H, C)  
        x_dst = x_src 

        x = (x_src, x_dst)

        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=self.fill_value, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' simultaneously is currently not yet supported for 'edge_index' in a 'SparseTensor' form"
                    )

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr, size=size)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        x_second_transformed = self.lin_second_graph(x_second).view(-1, H, C) 
        alpha_second = torch.einsum('bhd,nhd->bhn', x_src, x_second_transformed) 
        alpha_second = F.leaky_relu(alpha_second, self.negative_slope)
        maskbool = maskbool.unsqueeze(1).expand(-1, self.heads, -1)
        alpha_second = torch.softmax(alpha_second, dim=-1)  
        alpha_second =  F.dropout(alpha_second , p=self.dropout)
        alpha_second = alpha_second * maskbool
        cross = alpha_second
        x_second_graph_agg = torch.einsum('bhn,nhd->bhd', alpha_second, x_second_transformed) 

        if self.concat:
            x_second_graph_agg = x_second_graph_agg.reshape(-1, self.heads * self.out_channels)
        else:
            x_second_graph_agg = x_second_graph_agg.mean(dim=1)
        out = out + x_second_graph_agg

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha),cross
                else:
                    return out, (edge_index, alpha), cross 
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels}, heads={self.heads})'





class LogitGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4):
        super(LogitGCN, self).__init__()
        self.num_layers = num_layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.norm1 = BatchNorm(hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))
        
        # Output GCN layer
        self.conv_out = GCNConv(hidden_channels, out_channels)
        self.norm_out = BatchNorm(out_channels)
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            x = self.dropout(x)
            x += residual  
        x = self.conv_out(x, edge_index)
        x = self.norm_out(x)
        return x  # logits for Gumbel softmax



class Cross_GAT(nn.Module):

    def __init__(self):

        super(Cross_GAT, self).__init__()
        self.drug_layers = nn.ModuleList([GATConv(in_channels_first=-1, in_channels_second=1280, out_channels=96, heads=8, dropout=0.3, add_self_loops=False)])
        self.num_layers = 2 # set as per the need
        for i in range(self.num_layers):
            self.drug_layers.append(GATConv(in_channels_first=96 * 8, in_channels_second=1280, out_channels=96, heads=8, dropout=0.3, add_self_loops=False))
        self.prot_layers = nn.ModuleList([GATConv(in_channels_first=-1, in_channels_second=768, out_channels=160, heads=8, dropout=0.2, add_self_loops=True)])
        for i in range(self.num_layers):
            self.prot_layers.append(GATConv(in_channels_first= 160 * 8, in_channels_second=768, out_channels=160, heads=8, dropout=0.2, add_self_loops=True))
        self.act = nn.ReLU()
        self.gcn1 = LogitGCN(in_channels=-1, hidden_channels=256, out_channels=2)  
        self.gcn2 = LogitGCN(in_channels=-1, hidden_channels=256, out_channels=2)  

    def forward(self, x_drug, edge_index_drug, x_protein, edge_index_protein):

        attention_weights = []
        selected_percentages = []  

        for i in range(self.num_layers):

            logits1 = self.gcn1(x_drug, edge_index_drug)  
            logits2 = self.gcn2(x_protein, edge_index_protein) 
            m1 = F.gumbel_softmax(logits1, tau=0.5, hard=True)[:, 0] 
            m2 = F.gumbel_softmax(logits2, tau=0.5, hard=True)[:, 0] 
            mask = torch.outer(m1, m2) 
            mask = torch.ones_like(mask)
            x_drug, (alpha_drug, _), alpha_second_drug = self.drug_layers[i](
                x_drug, edge_index_drug, x_protein, maskbool=mask, return_attention_weights=True
            )
            x_protein, (alpha_protein, _), alpha_second_protein = self.prot_layers[i](
                x_protein, edge_index_protein, x_drug, maskbool=mask.T, return_attention_weights=True
            )
            attention_weights.append({
                'alpha_drug': alpha_drug,
                'alpha_second_drug': alpha_second_drug,
                'alpha_protein': alpha_protein,
                'alpha_second_protein': alpha_second_protein
            })
        return x_drug, x_protein, attention_weights , mask , mask.T 





class Net(pl.LightningModule):
    def __init__(self, cfg,dataset,network,optimizer,criterion,GAT_params , fold_idx=None):
        super().__init__()

        self.fold_idx = fold_idx
        self.cross_gat = Cross_GAT()
        self.drug_attn = []
        self.prot_attn = []
        self.nodes = 0
        drug_gat_dim = GAT_params['drug_gat']['out_channels']*GAT_params['drug_gat']['heads']
        prot_gat_dim = GAT_params['prot_gat']['out_channels']*GAT_params['prot_gat']['heads']
        if cfg['module']['GAT_params']['concat']['concat']:
            layers = [network['drug_dim']+network['prot_dim']+drug_gat_dim+prot_gat_dim]+network['layers']+[network['output_dim']]
        else:
            layers = [network['drug_dim']+network['prot_dim']] + network['layers'] + [network['output_dim']]
        layers = list(layers)
        self.model = MLP(layers, dropout=network['dropout'])
        self.prot_proj = nn.Linear(network['prot_dim'], network['drug_dim'])
        self.drug_proj = nn.Linear(network['drug_dim'], network['prot_dim'])

        self.cfg = cfg
        self.optimizer = optimizer
        if cfg['preprocess']['data_path'].split('/')[1] != 'warm_start_1_1':
            pos_weight = 1
        else:
            pos_weight = 1
        self.criterion: torch.nn = hydra.utils.instantiate(criterion,pos_weight=torch.tensor(pos_weight).float())
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_auc = 0
        self.test_auprc = 0
        self.test_f1 = 0
        self.save_hyperparameters(cfg)

    
    def forward(self, x1,x2,x1_org,x2_org,x1_network,x2_network,inv_drug,inv_target ,y ):

        x1 , x2 , attn , maskbool_drug_to_prot , maskbool_prot_to_drug  = self.cross_gat( x1, x1_network , x2, x2_network )
        maskbool_drug_to_prot = update_maskbool(maskbool_drug_to_prot, inv_drug, inv_target, y) 
        maskbool_prot_to_drug = update_maskbool(maskbool_prot_to_drug, inv_target, inv_drug, y) 
        contrastive_loss_drug_to_prot = contrastive_loss(x1, self.prot_proj(x2), maskbool_drug_to_prot)
        contrastive_loss_prot_to_drug = contrastive_loss(x2, self.drug_proj(x1), maskbool_prot_to_drug)
        total_contrastive_loss = contrastive_loss_drug_to_prot + contrastive_loss_prot_to_drug

        x1 = x1[inv_drug]
        x2 = x2[inv_target]


        p1 = x1
        p2 = x2

        if self.cfg['module']['GAT_params']['concat']['concat']:

            x1 = torch.cat([x1,x1_org],dim=1)
            x2 = torch.cat([x2,x2_org],dim=1)
           
            
        else:
            alpha = self.cfg['module']['GAT_params']['concat']['alpha']
            x1 = x1+(x1_org*alpha)
            x2 = x2+(x2_org*alpha)
    
        data = torch.cat([x1,x2],dim=1) 
        scores = self.model(data)

        return torch.squeeze(scores, dim=1) , data,  total_contrastive_loss 



    def training_step(self, batch, batch_idx):
        x1,x2,y, drugs, targets = batch

        x1_org , x2_org = x1, x2

        x1,x2,x1_network,x2_network, inv_drug, inv_target = self.common_preprocess(x1,x2,drugs,targets ,batch_idx)

        loss, scores = self.common_step(x1,x2,x1_org,x2_org,x1_network,x2_network, y, inv_drug, inv_target, batch_idx)

        self.training_step_outputs.append({"loss":loss, "scores":scores, "y":y})
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):    
        train_loss, train_auc = self.common_epoch_end(self.training_step_outputs)
        self.training_step_outputs.clear()
        self.log_dict({"train_loss":train_loss, "train_auc":train_auc})
        
    def validation_step(self, batch, batch_idx):
        x1,x2,y, drugs, targets = batch
        x1_org , x2_org = x1, x2

        x1,x2,x1_network,x2_network, inv_drug, inv_target = self.common_preprocess(x1,x2,drugs,targets ,batch_idx)


        loss, scores = self.common_step(x1,x2,x1_org,x2_org,x1_network,x2_network, y, inv_drug, inv_target, batch_idx)
        self.validation_step_outputs.append({"loss":loss, "scores":scores, "y":y})
        return loss

    def on_validation_epoch_end(self):
        val_loss, val_auc = self.common_epoch_end(self.validation_step_outputs)
        self.validation_step_outputs.clear()
        self.log_dict({"val_loss":val_loss, "val_auc":val_auc}, prog_bar=True)
  
    def test_step(self, batch, batch_idx):
        x1,x2,y, drugs, targets = batch
        x1_org , x2_org = x1, x2

        x1,x2,x1_network,x2_network, inv_drug, inv_target = self.common_preprocess(x1,x2,drugs,targets ,batch_idx)

        loss, scores = self.common_step(x1,x2,x1_org,x2_org,x1_network,x2_network, y, inv_drug, inv_target, batch_idx)
        self.test_step_outputs.append({
            "loss": loss,
            "scores": scores,
            "y": y,
            "drugs": inv_drug,  
            "targets": inv_target ,
            # "gdrugs": x1_network,  
            # "gtargets": x2_network
        })

    
        return loss


    def on_test_epoch_end(self):
        test_loss, test_auc, test_auprc, test_bcm, test_f1 = self.for_test_epoch(self.test_step_outputs)
        print(test_bcm)
        self.test_auc, self.test_auprc, self.test_f1 = test_auc, test_auprc, test_f1
        #self.test_step_outputs.clear()
        self.log_dict({"test_auc":test_auc, "test_auprc":test_auprc, "test_f1":test_f1})

    def common_step(self, x1,x2,x1_org,x2_org,x1_network,x2_network, y, inv_drug, inv_target, batch_idx):     
        scores, _ , CL  = self.forward(x1, x2, x1_org, x2_org, x1_network, x2_network, inv_drug, inv_target ,y ) 
        loss =  self.criterion(scores, y)  + CL
        return loss, scores


    def common_preprocess(self, x1,x2,drugs,targets ,batch_idx):

       
        mapping = {}
        for index, value in enumerate(drugs):
            if value.item() not in mapping.keys():
                mapping[value.item()] = index
        drugs = torch.tensor([mapping[value.item()] for value in drugs])
        
        mapping = {}
        for index, value in enumerate(targets):
            if value.item() not in mapping.keys():
                mapping[value.item()] = index
        targets = torch.tensor([mapping[value.item()] for value in targets])
        
        drug_index, inv_drug = torch.unique(drugs,return_inverse=True)
        target_index, inv_target = torch.unique(targets,return_inverse=True)
        x1 = x1[drug_index]
        x2 = x2[target_index]

   
        
        x1_network = torch.cdist(x1,x1,p=2)


        x1_network[x1_network<self.cfg['module']['GAT_params']['drug_gat']['threshold']] = 1
        x1_network[x1_network>=self.cfg['module']['GAT_params']['drug_gat']['threshold']] = 0
        x1_network = torch.triu(x1_network, diagonal=1)
        
        x2_network = torch.cdist(x2,x2,p=2)


        x2_network[x2_network<self.cfg['module']['GAT_params']['prot_gat']['threshold']] = 1
        x2_network[x2_network>=self.cfg['module']['GAT_params']['prot_gat']['threshold']] = 0
        x2_network = torch.triu(x2_network, diagonal=1)
        

    
        x1_network = x1_network.nonzero(as_tuple=False)
        x2_network = x2_network.nonzero(as_tuple=False)
        x1_network = x1_network.t()
        x2_network = x2_network.t()


        return x1,x2,x1_network,x2_network, inv_drug, inv_target


    def common_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        scores = torch.cat([x["scores"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])
        metric1 = torchmetrics.classification.BinaryAUROC(thresholds = None)
        auc = metric1(scores, y)
        return avg_loss, auc
    
    def for_test_epoch(self, outputs):          
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        scores = torch.cat([x["scores"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])

        metric1 = torchmetrics.classification.BinaryAUROC(thresholds = None)
        auc = metric1(scores, y)
        metric2 = torchmetrics.classification.BinaryAveragePrecision(thresholds = None)
        auprc = metric2(scores, y.long())
        metric3 = torchmetrics.classification.BinaryConfusionMatrix(threshold=0.5).to(device)
        bcm = metric3(scores.to(device), y.to(device))
        metric4 = torchmetrics.classification.BinaryF1Score(threshold=0.5).to(device)
        f1 = metric4(scores.to(device), y.to(device))
        return avg_loss, auc, auprc, bcm, f1

    def configure_optimizers(self):
        if self.optimizer['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer['lr'], weight_decay=self.optimizer['weight_decay'])
        elif self.optimizer['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.optimizer['lr'], weight_decay=self.optimizer['weight_decay'],momentum = 0.9)
        elif self.optimizer['optimizer'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.optimizer['lr'], weight_decay=self.optimizer['weight_decay'], momentum=0.9)
        else:
            print("optimizer not recognized")
            sys.exit()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'val_loss'}

         
