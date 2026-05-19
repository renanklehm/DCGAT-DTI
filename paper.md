Bioinformatics Advances, 2026, 00, vbaf306
https://doi.org/10.1093/bioadv/vbaf306
Advance Access Publication Date: 15 December 2025
Original Article

Systems  biology

DCGAT-DTI: dynamic cross-graph attention network for
drug–target interaction prediction
Abrar Rahman Abir1 , Muhtasim Noor Alif2, Wencai Zhang3, Khandakar Tanvir Ahmed2,�,
Wei Zhang2,�
1Department of Computer Science and Engineering, Bangladesh University of Engineering and Technology, Dhaka 1000, Bangladesh
2Department of Computer Science, University of Central Florida, Orlando, FL 32816, United States
3Division of Cancer Research, Burnett School of Biomedical Sciences, University of Central Florida, Orlando, FL 32827, United States
�Corresponding author. Wei Zhang, Department of Computer Science, University of Central Florida, 4000 Central Florida Blvd., Orlando, FL 32816, United
States. E-mail:  wzhang.cs@ucf.edu; Khandakar Tanvir Ahmed, Department of Computer Science, University of Central Florida, 4000 Central Florida Blvd.,
Orlando, FL 32816, United States. E-mail:  khandakar.tanvir.ahmed@ucf.edu.
Associate Editor: Travis Wheeler

Abstract

Motivation: Drug–target interaction (DTI) prediction accelerates drug discovery by identifying interactions between chemical compounds and
proteins. Existing methods often rely on drug-drug and protein-protein similarity graphs but process them independently, limiting their ability to
model  interdependencies  between  modalities.  Moving  beyond  isolated  embedding  generation  from  protein  and  drug  graphs,  we  propose
DCGAT-DTI, a novel deep learning framework with a dynamic cross-graph attention (DCGAT) module that dynamically models intra- and cross-
graph interactions. Initial embeddings are generated using pretrained language models. Similarity graphs constructed from these embeddings
are passed to DCGAT, which uses a Graph Convolutional Network-based Cross-Neighborhood Selection network to dynamically select cross-
modal neighbors. This allows drug and protein embeddings to incorporate information from both modalities through intra- and cross-graph atten-
tion mechanisms.

Results: Extensive evaluations on four benchmark datasets demonstrate that DCGAT-DTI outperforms state-of-the-art methods across warm
and cold start splits for both balanced and unbalanced datasets. In the challenging unbalanced cold start scenarios, it achieves significant im-
provement in performance for both drugs and proteins over the baselines.

Availability and implementation: Source code is available at https://github.com/compbiolabucf/DCGAT-DTI.

1 Introduction
Accurate prediction of drug–target interactions (DTIs) plays
a crucial role in modern drug discovery, where understanding
how  pharmaceutical  compounds  interact  with  proteins,
enzymes, and receptors can streamline and de-risk the tradi-
tionally  lengthy  and  expensive  development  process  (Zheng
et  al.  2020).  Beyond  identifying  prospective  binders,  DTI
insights  are  vital  for  uncovering  potential  side  effects,  en-
abling drug repurposing, and optimizing lead compounds for
improved therapeutic efficacy. As such, DTIs remain central
to precision medicine and continue to guide a wide range of
computational strategies aimed at enhancing pharmaceutical
research and development.

Over  the  years,  these  computational  strategies  have
evolved from classical machine learning methods to sophisti-
cated deep learning architectures. Early approaches often re-
lied  on  quantitative  structure–activity  relationship  (QSAR)
techniques,  using  Support  Vector  Machines  or  Random
Forests
to  process  hand-crafted  molecular  descriptors
(Faulon et al. 2008, Ballester and Mitchell 2010). While these
methods were valuable for initial screening, they struggled to
capture  the  subtle  factors  underlying  complex  molecular

interactions,  particularly  when  the  number  of  known  drugs
for  a  target  protein  was  limited  (Yamanishi  et  al.  2010).
Subsequent work leveraged docking-based techniques, which
assess binding energetics from crystallographic or homology-
modeled  structures  (Rarey  et  al.  1996),  and  more  recent
structure-based  deep  learning  approaches  have  further  ad-
vanced  this  direction  by  integrating  three-dimensional  (3D)
atomic representations with higher fidelity (Wu et al. 2024).
Although more expressive, these methods are constrained by
the  scarcity  of  high-quality  protein  structure  data.  In  con-
trast,  sequence-based  approaches  sidestep  the  need  for  de-
tailed  3D  structures  by  encoding  molecules  as  SMILES  and
proteins as  amino  acid sequences  (Chen et  al. 2020, Huang
et al. 2021), thus broadening the range of tractable targets.

strategy

With the advent of deep learning, automated feature extraction
for  DTI  prediction.
emerged  as  a  powerful
Convolutional and recurrent neural networks excelled at captur-
ing  local  and  sequential  patterns,  respectively  ( €Ozt€urk  et  al.
2018,  Lee  et  al.  2019,  Wang  et  al.  2020).  Graph  Neural
Networks (GNNs) then extended this capability by treating mol-
ecules as atom-level graphs, preserving important topological in-
formation  (Wei  et  al.  2016,  2022).  Meanwhile,  knowledge
graph–driven  approaches  integrate  diverse  biomedical  entities—

Received: July 31, 2025; Revised: November 6, 2025; Accepted: November 13, 2025
© The Author(s) 2025. Published by Oxford University Press.
This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which
permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.

l

D
o
w
n
o
a
d
e
d

f
r
o
m
h

t
t

p
s
:
/
/

i

a
c
a
d
e
m
c
.
o
u
p
.
c
o
m
b
o
n
o
r
m
a

/

f

i

i

l

/

/

t
i
c
s
a
d
v
a
n
c
e
s
/
a
r
t
i
c
e
6
1
/
v
b
a
3
0
6
8
3
8
0
3
4
9
b
y
g
u
e
s
t

f

/

o
n
1
9
M
a
y
2
0
2
6

2

Abir et al.

compounds,  proteins,  diseases—into  unified  relational  frame-
works, though often at a higher computational cost (Zhang et al.
2017,  Ahmed  et  al.  2020,  Thafar  et  al.  2020,  Ye  et  al.  2021).
The  incorporation  of  attention-based  models  and  transformers,
adapted from natural language processing, further elevated pre-
dictive  accuracy  by  capturing  long-range  dependencies  in
SMILES  and  protein  sequences  (Kalakoti  et  al.  2022,  Ahmed
et al. 2024). In particular, pretrained language models specialized
for proteins, such as ProtT5 from the ProtTrans family (Elnaggar
et al. 2022) and ProteinBERT (Brandes et al. 2022), or for chemi-
cal tokens such as ChemBERTa (Chithrananda et al. 2020), pro-
vide rich embeddings that enhance downstream tasks.

Despite  these  advances,  many  current  approaches  handle
drug and protein embeddings in isolation, focusing primarily on
local  structures  or  sequences  within  each  modality.  By  relying
on separate feature-learning pipelines and late-stage fusion, they
often overlook the intricate cross-modal interactions that under-
pin  binding  mechanisms  (Nguyen  et  al.  2021,  Ahmed  et  al.
2024). Consequently, these models underutilize crucial informa-
tion that might emerge when drug and protein embeddings co-
evolve  during  training.  This  gap  points  to  an  urgent  need  for
approaches  that  explicitly  integrate  cross-modal  dependencies
throughout the representation process, forming a more holistic
and robust foundation for DTI prediction.

In  this  work,  moving  beyond  isolated  embedding  genera-
tion  from  protein  and  drug  graphs,  we  introduce  DCGAT-
DTI,  a  framework  designed  to  address  the  limitations  of
current approaches by explicitly capturing cross-modal rela-
tionships  between  drugs  and  proteins.  Our  method  begins
with  state-of-the-art  language  models,  ESM-2  for  protein
sequences and ChemBERTa for SMILES, to provide rich ini-
tial embeddings. We then construct modality-specific similar-
ity graphs to preserve neighborhood information within each
domain. Central to our approach is a Dynamic Cross-Graph
Attention Network (DCGAT), which dynamically selects rel-
evant cross-modal neighbors at each layer, allowing drug and
protein representations to mutually inform and refine one an-
other. This is further strengthened by a dual-objective train-
ing  strategy:  a  binary  cross-entropy  (BCE)  loss  enhances
interaction  prediction  accuracy,  while  a  supervised  contras-
tive loss differentiates true from false interactions in the latent
space.  Through  extensive  experiments  on  four  benchmark
datasets, we demonstrate that DCGAT-DTI can significantly
enhance DTI modeling by integrating intra-graph and cross-
graph  attention,  offering  a  comprehensive  framework  that
advances the state of the art in drug discovery.

2 Method
In this section, we first introduce the mathematical notations
utilized in this study. Next, a comprehensive overview of the
proposed  framework,  DCGAT-DTI,  is  presented,  followed
by a detailed explanation of the methodology. Finally, we de-
scribe the baseline models employed to highlight the improve-
ments achieved by our approach.

2.1 Overview of the framework
We  introduce  DCGAT  Network  to  address  the  limitations  of
traditional DTI prediction, which relies on hand-crafted features
and  struggles  with  novel  compounds  and  targets.  While  lan-
guage  models  capture  biochemical  patterns  in  protein  and
SMILES  sequences  (Kalakoti  et  al.  2022),  DTI  complexity
requires modeling cross-modal dependencies within protein and

chemical interaction networks. For this reason, DCGAT jointly
processes drug and protein similarity graphs by capturing their
underlying  patterns  and  dynamically  attending  to  cross-modal
neighborhoods.  This  joint  processing  enables  modeling  both
intra-graph relationships and cross-graph interactions, allowing
the embeddings of drugs and proteins to mutually inform each
other for better representation learning.

Our  framework  begins  with  state-of-the-art  protein  and
chemical  language  models,  ESM-2  Lin  et  al.  (2023a) and
ChemBERTa  Chithrananda  et  al.  (2020),  to  generate  rich
contextual  embeddings  for  protein  sequences  and  SMILES
representations,  respectively.  These  embeddings  serve  as  in-
put  to  the  DCGAT  module,  where  we  construct  similarity
graphs for both drugs and proteins. Graphs are constructed
by  defining  edges  between  nodes  if  their  pairwise  similarity
exceeds a predefined threshold, capturing local neighborhood
relationships. The DCGAT module then dynamically selects
cross-graph  neighborhood  for  both  modalities  at  each  layer
using  a  Cross-Neighborhood  Selection  (CNS)  network,
implemented  as  Graph  Convolutional  Network  (GCN).  By
leveraging  intra-graph  attention  to  capture  relationships
within each graph, and cross-graph attention to learn interac-
tions  between  drugs  and  proteins,  DCGAT  generates
information-rich  embeddings  for  both  drugs  and  proteins.
These  embeddings  are  passed  through  a  multilayer  percep-
tron (MLP) to predict the interaction.

We  employ  a  dual-objective  training  strategy  where  BCE
loss guides the model in accurately predicting the presence or
absence  of  interactions,  while  a  supervised  contrastive  loss
aligns the embeddings in the latent space, pulling interacting
pairs  closer  and  pushing  non-interacting  pairs  apart.  This
combined loss framework enhances the model’s ability to dis-
criminate between true and false interactions, ensuring robust
and  generalizable  representations.  Figure  1 illustrates  the
overall workflow of DCGAT-DTI and the notations used to
define the proposed model are summarized in Table 1.

from

dataset,

2.2 Protein and drug sequence encoding
The field of protein sequence analysis has witnessed remark-
able progress with the emergence of ESM-2, a state-of-the-art
language model that excels in capturing the intricate patterns
within protein structures (Lin et al. 2023). DCGAT-DTI lev-
erages  this  protein  language  model,  which  generates  rich
1280-dimensional embeddings through its architecture of 33
layers and 650 million parameters. ESM-2 is trained on the
UniRef50
the  UniProt
extracted
Knowledgebase (The UniProt Consortium 2023). While other
such  as  MSA-based  models
contemporary  models,
[AlphaFold3,  RoseTTAFold  (Baek  et  al.  2021,  Abramson
et  al.  2024)]  and  transformer-based  models  [ProtBert
(Elnaggar et al. 2022)], offer impressive capabilities, ESM-2
provides  a  superior  balance  of  efficiency  and  accuracy.
Particularly  noteworthy  is  its  performance  advantage  over
MSA-dependent  models,  processing  sequences  up  to  100×
faster while achieving comparable accuracy levels in DTI pre-
dictions (Kalakoti et al. 2022, Lin et al. 2023). This combina-
tion of speed and analytical depth enables ESM-2 to excel in
detecting  both  fine-grained  sequence  motifs  and  extensive
structural  relationships,  positioning  it  as  an  invaluable  tool
for understanding protein behavior and interactions.

ChemBERTa  (Chithrananda  et  al.  2020)  represents  a  sig-
nificant  advancement  in  molecular  representation  learning,
specifically  designed  to  process  SMILES  sequences  for  drug

l

D
o
w
n
o
a
d
e
d

f
r
o
m
h

t
t

p
s
:
/
/

i

a
c
a
d
e
m
c
.
o
u
p
.
c
o
m
b
o
n
o
r
m
a

/

f

i

i

l

/

/

t
i
c
s
a
d
v
a
n
c
e
s
/
a
r
t
i
c
e
6
1
/
v
b
a
3
0
6
8
3
8
0
3
4
9
b
y
g
u
e
s
t

f

/

o
n
1
9
M
a
y
2
0
2
6

DCGAT-DTI

3

l

D
o
w
n
o
a
d
e
d

f
r
o
m
h

t
t

p
s
:
/
/

Figure 1. Overview of the DCGAT-DTI framework. (a) DCGAT-DTI framework. Initial encoding for proteins and drugs are obtained by language models
from protein sequences and drug SMILES sequences. Then protein and drug similarity matrices construct similarity graphs which are processed by
DCGAT module to generate mutual information rich embeddings. The concatenated embeddings are passed through the MLP to give the DTI prediction.
(b) In DCGAT module, cross-model neighbors for a node (say, p1) are selected dynamically in each layer by the CNS network. Next, the updated
embedding is obtained by aggregating the embeddings of its intra-graph and cross-graph neighbors. See Equation (6) for aggregation function.

Table 1. Table of notations.

Name

Definition

p, q, m, n

Protein encoding size, drug encoding size, number

X 2 Rm × p
Y 2 Rn × q
P 2 Rm × p
D 2 Rn × q
GP ¼ ðVP; EPÞ
GD ¼ ðVD; EDÞ
τP, τD
SP, SD

of proteins, number of drugs

Protein sequence embedding by ESM-2
Drug SMILES embedding by ChemBERTa
Protein embedding from DCGAT module
Drug embedding from DCGAT module
Protein similarity graph
Drug similarity graph
Protein and drug similarity threshold
Euclidean distance matrices

transformer-based

discovery applications. The model builds upon the BERT ar-
chitecture,  employing  a
framework
adapted  for  chemical  structure  understanding  through
masked language modeling on large-scale chemical databases.
In this work, we use a widely adopted ChemBERTa version
trained on 10 million SMILES sequences from the PubChem
database  (Kim  et  al.  2023).  For  DTI  prediction  tasks,
ChemBERTa excels at capturing both local chemical patterns
and global molecular features by treating SMILES strings as a
specialized  chemical  language.  The  learned  representations
preserve crucial information about molecular properties and
structural motifs, which are essential for predicting potential
interactions with protein targets.

2.3 Protein and drug graph construction
Let  X ¼ ½x1; x2; . . .; xm�T 2 Rm × p  represent  the  protein  embed-
ding matrix, where each column vector xi 2 Rp  denotes the p-
dimensional encoding generated by ESM-2 from the amino acid
sequence of the ith protein. To capture the dual-neighborhood,
we  construct  a  protein  similarity  graph  GP ¼ ðVP; EPÞ,  where
VP ¼ fv1; v2; . . . ; vmg represents the set of nodes corresponding
to  the  m  proteins,  and  EP � VP × VP  denotes  the  set  of  edges

connecting  similar  proteins.  The  edge  set  EP  is  determined  by
computing a pairwise Euclidean distance matrix SP 2 Rm × m  be-
tween protein encodings. An edge exists between proteins i and
k if their distance is below a threshold τP, formally defined as
EP ¼ ði; kÞ j SPði; kÞ ≤ τP.  Similarly,  for  drugs,  we  have  Y ¼
½y1; y2; . . .; yn�T 2 Rn × q  and  construct  GD ¼ ðVD; EDÞ where
VD ¼ fv1; v2; . . . ; vng and  ED ¼ ðj; lÞ j SDðj; lÞ ≤ τD,  with  SD 2
Rn × n  being the pairwise distance matrix for drug encodings.

2.4 DCGAT network
The DCGAT Network takes the protein graph GP ¼ ðVP; EPÞ
and  drug  graph  GD ¼ ðVD; EDÞ,  along  with  their  respective
node  features  X  and  Y  as  inputs.  The  primary  goal  of
DCGAT is to dynamically learn the interactions between the
two  graphs  by  simultaneously  attending  to  both  graphs,
updating  each  node’s  embedding based  on  both  intra-graph
and cross-graph information. This dual-attention mechanism
enables  the  model  to  capture  complex  relationships  within
each modality and across the drug-protein interaction space.

2.4.1 Dynamic cross-graph neighborhood selection
Within DCGAT, at each layer ‘, we employ CNS networks to
dynamically determine cross-graph neighbors. For each drug
node j 2 VD, the CNS network fd  generates a probability dis-
tribution  over  all  protein  nodes  in  VP,  determining  whether
to  select  each  protein  node  as  a  neighbor  or  not.  Similarly,
for each protein node i 2 VP, another CNS network fp  com-
putes a probability distribution over all drug nodes in VD  for
neighbor selection. Both CNS networks are implemented us-
ing GCNs to leverage the structural information encoded in
GD  and GP.

The  probability  distribution  for  protein  node  i  over  drug

nodes is given by:

i

a
c
a
d
e
m
c
.
o
u
p
.
c
o
m
b
o
n
o
r
m
a

/

f

i

i

l

/

/

t
i
c
s
a
d
v
a
n
c
e
s
/
a
r
t
i
c
e
6
1
/
v
b
a
3
0
6
8
3
8
0
3
4
9
b
y
g
u
e
s
t

f

/

o
n
1
9
M
a
y
2
0
2
6

4

Abir et al.

i ¼ fpðp‘
ρ‘

i ; D‘Þ 8i 2 VP

(1)

i 2 Rn × 2  is a probability distribution over two actions
where ρ‘
(select or not select) for each drug node. Here, p‘
i  denotes the
embedding of protein i and D‘ represents the embeddings of
all drugs at layer ‘. At the first layer (‘ ¼ 1), D1 ¼ Y and p1
i  is
the ith row of X. Similarly, for each drug node j, the CNS net-
work fd  produces:

j ¼ fdðd‘
ρ‘

j ; P‘Þ 8j 2 VD

(2)

j 2 Rm × 2  represents the probability distribution over
where ρ‘
selecting each protein node and P‘ represents the embeddings
of all proteins at layer ‘.

For each drug node j 2 VD, an action vector a‘

j 2 f0; 1gm  is
sampled  from  the  policy  distribution  ρj  using  the  Straight-
through Gumbel-softmax (GS) estimator following Finkelshtein
et al. (2023). The resulting binary vector aj  indicates which pro-
tein nodes are neighbors of the drug node j. This process can be
expressed as: aj � ρj; 8j 2 VD. Similarly, for each protein node
i 2 VP, the CNS network generates a binary action vector a‘
i 2
f0; 1gn  sampled as ai � ρi; 8i 2 VP. The sampled binary vectors
ai  and aj  are used to construct the sets N P  and N D, which de-
fine
for  cross-graph  attention.
Specifically, N PðiÞ represents the set of selected drug neighbors
for  each  protein  node  i,  while  N DðjÞ represents  the  set  of  se-
lected  protein  neighbors  for  each  drug  node  j.  Leveraging  the
GS  estimator  ensures  that  the  neighbor  selection  process
remains  differentiable,  enabling  end-to-end  model  training
through gradient-based optimization.

selected  neighbors

the

2.4.2 Straight-through Gumbel-softmax estimator
In our approach, we rely on a CNS network for dynamically
selecting  neighbors  in  the  cross-graph  attention  mechanism.
Since  selecting  neighbors  is  a  non-differentiable  process,  we
employ the GS estimator to provide a differentiable, continu-
ous approximation for this discrete selection. Given a proba-
bility  distribution  ρ  and  a  temperature  parameter  T  which
controls the sharpness of the probability distribution, the GS
scores are computed as:

GSðρ; TÞ ¼

exp

P

a2Ω exp

��

logðρÞ þ g

��

�

�

=T

�

�

logðρðaÞÞ þ gðaÞ

=T

(3)

where g � Gumbelð0; 1Þ is a Gumbel-distributed random var-
iable and Ω  is the set of actions: select or not select. During
the forward pass, the GS estimator mimics discrete sampling
by selecting a specific action, while in the backward pass, it
provides  a  continuous  approximation,  enabling  gradient-
based updates for end-to-end model training.

2.4.3 Intra-graph and cross-graph attention
The DCGAT module employs two types of multi-head atten-
tion mechanisms to update node embeddings: intra-graph at-
tention and cross-graph attention. The intra-graph attention
mechanism refines the embeddings of nodes within the same
graph  by  attending  to  their  neighbors.  For  a  protein  node
i 2 VP,  the  attention  coefficient  αik  between  node  i  and  its
neighbor k 2 EPðiÞ is given by:

�
LeakyReLUðϕ>½WpikWpk�Þ

�

exp

P

r2EPðiÞ exp

�

�
LeakyReLUðϕ>½WpikWpr�Þ

αik ¼

(4)

where  W  and  ϕ  are  the  learnable  weights  for  intra-graph
aggregation,.

In  addition  to  attending  to  intra-graph  neighbors,  the
DCGAT  module  incorporates  cross-graph  attention  to  dy-
namically  selected  cross-graph  neighbors  from  the  other
graph. For a protein node i, the cross-graph attention coeffi-
cient βij  between protein node i and its selected drug neigh-
bors j 2 N PðiÞ is computed as:

�

exp

βij ¼

P

r2N PðiÞ exp

�

LeakyReLU ðϕ>

cr½W crpikW crdj�Þ

LeakyReLU ðϕ>

cr½W crpikW crdr�Þ

�

�

(5)

where W cr  and ϕcr  are the learnable weights for cross-graph
aggregation.  These  dual  attention  mechanisms  enable
DCGAT to capture both local interactions within each graph
and  the  dynamic  interconnections  across  graphs,  providing
enriched embeddings for downstream tasks. Intra-graph and
cross-graph attention scores are calculated for each drug fol-
lowing similar procedures.

Finally,  the embeddings for each protein node  i and drug

node j at layer l are updated as follows:

X

p‘ þ 1
i

¼ σ

irW ‘
α‘

1p‘

r þ

X

irW ‘
β‘

2d‘

r

d‘ þ 1
j

¼ σ

r2EPðiÞ
X

r2EDðjÞ

α‘
jr

cW ‘

1d‘

r þ

r2N PðiÞ
X

r2N DðjÞ

β‘
ir

cW ‘

2p‘

r

!

!

(6)

(7)

j

i

and d‘ þ 1

where σ  is nonlinear activation function. p‘ þ 1
rep-
resent the updated node representations for the protein node
i  and  drug  node  j  after  a  single  attention  head.  To  enhance
model performance, we employ a multi-head attention mech-
anism.  This  approach  allows  the model  to  capture  different
aspects of the node interactions by attending to multiple sub-
spaces simultaneously, improving the ability to learn complex
relationships between nodes. Additionally, multi-head atten-
tion  stabilizes  training  by  reducing  the  sensitivity  of  the
model  to  initial  conditions  and  helps  prevent  overfitting  by
allowing  the  model  to  aggregate  information  from  different
attention heads.

The  final  embeddings  for  the  protein  and  drug  nodes  are
obtained by taking the mean of the embeddings from all H at-
tention heads. By dynamically selecting relevant cross-graph
neighbors  at  each  layer,  DCGAT  effectively  captures  the
complex interactions between drug and protein nodes, allow-
ing for a more expressive  and informative representation of
each node. This dual attention mechanism enables the model
to  incorporate  both  local  neighborhood  information  and
cross-graph  relationships.  As  a  result,  the  embeddings  are
enriched  with  comprehensive  context,  making  them  highly
informative and accurately reflective of the underlying drug-
protein interactions.

2.5 DTI prediction
After obtaining the updated drug embeddings D and protein
embeddings  P  from  the  DCGAT  module,  these  embeddings
are added with the initial drug and protein encodings Y  and

l

D
o
w
n
o
a
d
e
d

f
r
o
m
h

t
t

p
s
:
/
/

i

a
c
a
d
e
m
c
.
o
u
p
.
c
o
m
b
o
n
o
r
m
a

/

f

i

i

l

/

/

t
i
c
s
a
d
v
a
n
c
e
s
/
a
r
t
i
c
e
6
1
/
v
b
a
3
0
6
8
3
8
0
3
4
9
b
y
g
u
e
s
t

f

/

o
n
1
9
M
a
y
2
0
2
6

DCGAT-DTI

5

X,  respectively,  to  create  a  residual  connection.  For  each
drug-protein  pair,  the  concatenated  embedding  is  obtained
as follows:

Zij ¼ ½ðpi þ δ xiÞ ∥ ðdj þ γ yjÞ�

(8)

where Zij 2 Rm þ n  is the concatenated embedding for the pro-
tein node i and the drug node j. Here, δ and γ are hyperpara-
meters  controlling
residual
connections from the initial encodings. To simplify the nota-
tion, we refer to Zij  as Zk, where k indexes each drug-protein
pair in the batch.

the  contribution  of

the

We apply a supervised contrastive loss on the concatenated
embedding  Zk  based  on  the  DTI  label,  similar  to  the  NT-
Xent loss Khosla et al. (2020). For drug-protein pairs with an
interaction label yk ¼ 1, we pull their embeddings closer to-
gether,  while  for  pairs  with  an  interaction  label  yk ¼ 0,  we
push their  embeddings further apart.  The contrastive loss  is
then computed using the following equation:

Lcontrastive ¼ −

1
jPðkÞj

X

r2PðkÞ

log

P

exp ðZk � Zr=τÞ
l2AðkÞ exp ðZk � Zl=τÞ

(9)

where  τ  is  the  parameter  which  scales  similarity  scores  and
controls the sharpness of the softmax probability distribution
over pairwise similarities, influencing how strongly the model
distinguishes between positive and negative pairs. PðkÞ repre-
sents the set of positive samples (those drug-protein pairs that
share the same interaction label as the pair k), and AðkÞ rep-
resents  all  other  samples  in  the  batch  distinct  from  k.  The
concatenated embedding Zk  is then passed through an MLP
to predict the interaction score ~Ik  for each drug-protein pair:

~Ik ¼ MLPðZkÞ

(10)

where ~Ik  is the predicted interaction score for protein i and
drug j. The model is trained with a combination of BCE loss
for interaction prediction and the contrastive loss:

LBCE ¼ −

1
N

XN

k¼1

�
yk logð~IkÞ þ ð1 − ykÞ logð1 − ~IkÞ

�

(11)

where N is the total number of drug-protein pairs (samples)
in the batch. The total loss is a weighted combination of the
BCE loss and the contrastive loss:

Ltotal ¼ LBCE þ λ � Lcontrastive

(12)

where  λ  is  a  hyperparameter  balancing  the  contribution  of
the contrastive loss.

2.6 Baseline models
We evaluate the performance of DCGAT-DTI against several
state-of-the-art  models.  DTI-LM  leverages  language  models
to  generate  embeddings  from  protein  sequences  and  drug
SMILES,
for
context-aware  DTI  predictions  (Ahmed  et  al.  2024).  CCL-
DTI integrates multimodal features, including drug-drug and
protein-protein  interaction  networks,  using  an  attention-
based  fusion  mechanism  and  contrastive  loss  to  enhance
representation  learning  (Dehghan  et  al.  2024).  CAT-DTI

incorporating  graph  attention  networks

combines  CNNs  and  Transformers  with  cross-attention  to
capture DTIs while improving generalization through domain
adaptation  (Zeng  et  al.  2024).  TransDTI  utilizes  language
models to encode protein and drug sequences, followed by an
MLP that processes the language model outputs for DTI pre-
diction  (Kalakoti  et  al.  2022).  These  baselines  represent
diverse strategies for DTI prediction, against which DCGAT-
DTI  demonstrates  its  superior  ability  to  model  both  intra-
graph and cross-graph relationships.

3 Experiments
3.1 Datasets
The  proposed  framework  is  evaluated  on  four  widely-used
datasets: DrugBank (Knox et al. 2024), BindingDB (Liu et al.
2007),  Yamanishi_08  (Yamanishi  et  al.  2008),  and  Luo’s
dataset  (Luo  et  al.  2017).  The  DrugBank  and  BindingDB
datasets contain only protein and drug sequences with inter-
action labels. In contrast, the Yamanishi_08 and Luo’s data-
sets  include  both  protein  and  drug  sequences  as  well  as
heterogeneous knowledge graphs that provide additional in-
teraction information. The BindingDB dataset provides bind-
ing  affinity  (Kd)  values,  which  are  converted  into  binary
interaction labels using a predefined threshold to align with
the classification framework. This threshold ensures a consis-
tent DTI density across all datasets. Table 2 contains the sta-
tistics of the datasets.

3.2 Experimental setup
The  DrugBank  and  BindingDB  datasets  are  divided  into
training,  validation,  and  test  sets  with  ratios  of  0.79,  0.01,
and 0.20, respectively. This splitting follows three evaluation
strategies: warm start (the same drugs and proteins appear in
both  training  and  test  sets),  cold  start  for  drugs  (drugs  in
training and test sets are mutually exclusive), and cold start
for  proteins  (proteins  in  training  and  test  sets  are  mutually
exclusive).  The  Yamanishi_08  and  Luo’s  datasets  are
obtained from the source mentioned in (Ye et al. 2021) and
the same training and test splits as utilized in that study are
used to generate our results. All predictions are repeated 10×
with  different  splits,  reporting  the  mean  area  under  the
Receiver  Operating  Characteristic  curve  (AUROC)  and  the
area under the Precision-Recall curve (AUPRC). The experi-
ments are performed under two data settings: balanced data
with a  1:1  ratio of  positive  to  negative samples, and  unbal-
anced with a 1:10 ratio, or the maximum achievable ratio if
there  are  insufficient  negatives  to  meet  the  1:10  threshold.
The hyperparameters are reported in Supplementary Material
(Table 1, available as  supplementary data at Bioinformatics
Advances online).

3.3 Prediction performance
The  performance  evaluation  of  DCGAT-DTI  is  carried  out
under  all  three  different  splitting  scenarios,  including  warm
start, cold start for drug, and cold start for protein, with both

Table 2. Statistics of datasets.

Dataset

Proteins

Drugs

Interactions

DrugBank
BindingDB
Yamanishi_08
Luo’s

2203
879
722
1129

1603
9144
791
708

6041
4040
3448
1526

l

D
o
w
n
o
a
d
e
d

f
r
o
m
h

t
t

p
s
:
/
/

i

a
c
a
d
e
m
c
.
o
u
p
.
c
o
m
b
o
n
o
r
m
a

/

f

i

i

l

/

/

t
i
c
s
a
d
v
a
n
c
e
s
/
a
r
t
i
c
e
6
1
/
v
b
a
3
0
6
8
3
8
0
3
4
9
b
y
g
u
e
s
t

f

/

o
n
1
9
M
a
y
2
0
2
6

6

Abir et al.

Table 3. Performance on BindingDB dataset. Bold values indicate the best performance.

Scenario

Condition

DCGAT-DTI

DTI-LM

CCL-DTI

TransDTI

CAT-DTI

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

Balanced

Warm start
Cold start for drug
Cold start for protein

Unbalanced Warm start

Cold start for drug
Cold start for protein

0.943
0.875
0.838
0.945
0.888
0.871

0.938
0.889
0.809
0.841
0.751
0.582

0.939
0.872
0.812
0.945
0.895
0.831

0.934
0.879
0.787
0.839
0.744
0.463

0.890
0.869
0.767
0.900
0.820
0.803

0.879
0.860
0.745
0.783
0.675
0.438

0.926
0.870
0.806
0.941
0.872
0.818

0.918
0.878
0.779
0.834
0.708
0.456

0.902
0.855
0.783
0.913
0.864
0.817

0.899
0.852
0.769
0.809
0.712
0.445

Table 4. Performance on DrugBank dataset. Bold values indicate the best performance.

Scenario

Condition

DCGAT-DTI

DTI-LM

CCL-DTI

TransDTI

CAT-DTI

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

Balanced

Warm start
Cold start for drug
Cold start for protein

Unbalanced Warm start

Cold start for drug
Cold start for protein

0.968
0.915
0.933
0.974
0.902
0.944

0.959
0.903
0.940
0.887
0.697
0.839

0.951
0.902
0.923
0.960
0.890
0.938

0.953
0.899
0.935
0.863
0.674
0.821

0.927
0.886
0.882
0.951
0.844
0.876

0.916
0.851
0.890
0.860
0.650
0.793

0.934
0.877
0.916
0.952
0.876
0.916

0.935
0.889
0.920
0.858
0.651
0.789

0.949
0.893
0.905
0.953
0.865
0.910

0.933
0.876
0.923
0.850
0.661
0.826

Table 5. Performance on Yamanishi_08 dataset. Bold values indicate the best performance.

Scenario

Condition

DCGAT-DTI

DTI-LM

CCL-DTI

TransDTI

CAT-DTI

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

Warm start
Balanced
Unbalanced Warm start

Cold start for drug
Cold start for protein

0.988
0.989
0.814
0.932

0.970
0.942
0.517
0.769

0.974
0.984
0.785
0.911

0.966
0.930
0.451
0.739

0.913
0.947
0.729
0.893

0.892
0.877
0.398
0.712

0.969
0.984
0.762
0.902

0.961
0.927
0.442
0.729

0.952
0.954
0.759
0.918

0.939
0.900
0.437
0.740

Table 6. Performance on Luo’s dataset. Bold values indicate the best performance.

Scenario

Condition

DCGAT-DTI

DTI-LM

CCL-DTI

TransDTI

CAT-DTI

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

AUC

AUPRC

Balanced
Warm start
Unbalanced Warm start

Cold start for drug
Cold start for protein

0.953
0.981
0.789
0.860

0.949
0.920
0.504
0.647

0.944
0.971
0.760
0.832

0.948
0.906
0.393
0.595

0.883
0.959
0.706
0.796

0.870
0.897
0.381
0.500

0.938
0.971
0.742
0.823

0.939
0.902
0.383
0.589

0.916
0.966
0.729
0.828

0.899
0.905
0.377
0.575

balanced  and  unbalanced  data  settings.  Tables  3–6 summa-
rize the results of DCGAT-DTI in comparison with baseline
models,  including  DTI-LM  (Ahmed  et  al.  2024),  CCL-DTI
(Dehghan  et al. 2024), TransDTI (Kalakoti et al. 2022), and
CAT-DTI (Zeng et al. 2024). Our model consistently outper-
forms  all  baselines  across  all  datasets  and  scenarios,  demon-
strating the efficacy of leveraging dynamic cross-neighborhood
and contrastive learning in enhancing representation quality.

We observe that the cold start scenarios for drug and pro-
tein present significant challenge due to the inclusion of un-
seen entities during training. Among these, cold start for drug
emerges as the more challenging condition across all datasets,
attributed  to  the  structural  and  contextual  complexity  of
drugs,  which  makes  accurately  capturing  their  interactions
with proteins more difficult. Despite the inherent challenges

of  cold  start  conditions,  DCGAT-DTI  demonstrates  signifi-
cant  improvements  over  the  best-performing  baseline,  DTI-
LM,  in  both  drug  and  protein  scenarios.  For  cold  start  for
drug, the model achieves an average improvement of 2.32%
in  AUROC  and  11.12%  in  AUPRC  for  balanced  datasets,
and  2.02%  in  AUROC  and  11.81%  in  AUPRC  for  unbal-
anced datasets. Similarly, for cold start for protein, DCGAT-
DTI  achieves  2.49%  in  AUROC  and  4.03%  in  AUPRC  for
balanced  datasets,  and  2.78%  in  AUROC  and  10.17%  in
AUPRC  for  unbalanced  datasets.  These  consistent  trends
across  both  scenarios  highlight  DCGAT-DTI’s  strong  capa-
bility  to  capture  meaningful  DTIs  even  in  challenging  set-
tings.  Notably,  the  consistently  higher  AUPRC  gains  in
unbalanced  datasets  across  all  conditions  underscore  the
model’s robust ability to handle the more difficult prediction

l

D
o
w
n
o
a
d
e
d

f
r
o
m
h

t
t

p
s
:
/
/

i

a
c
a
d
e
m
c
.
o
u
p
.
c
o
m
b
o
n
o
r
m
a

/

f

i

i

l

/

/

t
i
c
s
a
d
v
a
n
c
e
s
/
a
r
t
i
c
e
6
1
/
v
b
a
3
0
6
8
3
8
0
3
4
9
b
y
g
u
e
s
t

f

/

o
n
1
9
M
a
y
2
0
2
6

DCGAT-DTI

7

tasks  involving  imbalanced  positive-to-negative  interaction
ratios. The largest AUPRC improvement of 11.81% for cold
start for drug and 10.17% for cold start for protein in unbal-
anced datasets further validate DCGAT-DTI’s capacity to ef-
fectively  mitigate  data  imbalance  and  enhance  interaction
prediction.  By  leveraging  its  dynamic  cross-neighborhood
mechanism,  DCGAT-DTI  adapts  to  the  complexities  of
sparse data, excelling in capturing relationships involving un-
seen  entities.  Similarly,  in  warm  start  splits,  DCGAT-DTI
exhibits  consistent  improvements  over  DTI-LM.  However,
DCGAT-DTI shows the highest average improvement across
all splitting and data settings in both AUROC (2.785%) and
AUPRC  (2.65%)  in  the  BindingDB  dataset,  over  the  best
baseline DTI-LM. Conversely, the lowest overall average im-
provement is found in the DrugBank dataset, with 2.04% in
AUROC and 2.08% in AUPRC. Overall, DCGAT-DTI’s con-
sistent and substantial improvements across all datasets and
conditions  validate  its  robust  design,  leveraging  dynamic
CNS and contrastive learning to capture meaningful relation-
ships. By achieving state-of-the-art performance across both
warm  and  cold  start  scenarios,  the  model  effectively
addresses  varying  levels  of  complexity  and  data  imbalance,
setting a new benchmark for DTI prediction.

3.4 Dynamic neighborhood selection
We investigate the effectiveness of the dynamic neighborhood
selection in DCGAT by analyzing the trade-off between the
percentage of selected nodes and AUROC, demonstrating the
impact  of  dynamic  CNS  on  model  performance.  Figure  2
illustrates  how  the  percentage  of  selected  nodes  affects
AUROC, influenced by the temperature parameter of the GS
estimator which controls the sharpness of the probability dis-
tribution.  The  percentage  of  selected  nodes  represents  the
proportion  of  protein  nodes  included  in  a  drug’s  cross-
neighborhood (and vice versa for proteins), calculated sepa-
rately for each modality and averaged across the dataset for a
comprehensive  measure.  We  conducted  this  experiment  on

BindingDB test set (balanced warm start). As the temperature
increases, the probability distribution becomes softer, result-
ing in more nodes being selected and increasing the percent-
age  of  selected  nodes.  The  dynamic  neighborhood  selection
plays a crucial role in DCGAT-DTI’s performance. Selecting
too  few  nodes  limits  the  model’s  ability  to  capture  diverse
and  meaningful  cross-graph  interactions,  whereas  selecting
too many nodes  introduces noise, diluting relevant relation-
ships.  This  trade-off  is  evident  in  Fig.  2,  where  the  model
achieves  peak  performance  at  an  optimal  percentage  of  se-
lected nodes, but AUROC drops significantly when the per-
centage deviates too far in either direction. This demonstrates
the importance of carefully balancing neighborhood selection
to effectively capture cross-modal dependencies.

3.5 Resilience to neighborhood noise
To  evaluate  the  robustness  of  DCGAT-DTI  to  noise  in  the
neighborhood embeddings, we conducted an experiment on the
BindingDB test set under the balanced warm start scenario. At
inference, we injected Gaussian noise with varying standard de-
viation  into  the  node  embeddings  for  both  intra-graph  and
cross-graph  neighborhoods  prior  to  aggregation.  The  model’s
predictive performance was then measured in terms of AUROC
for each noise level. As illustrated in Fig. 3, the AUROC remains
remarkably  stable  across  a  wide  range  of  noise  standard  devia-
tions, with only a slight decrease as noise increases. This demon-
strates that DCGAT-DTI is highly resilient to neighborhood-level
noise and does not exhibit excessive sensitivity to perturbations in
the node embeddings.

3.6 Prediction stability with respect to batch
neighborhood context
To  assess  whether  the  prediction  of  a  drug–target  pair  in
DCGAT-DTI is influenced by the other samples in the batch,
which means whether the neighborhood context within a test
batch  can  alter  prediction  outcomes,  we  conducted  a  batch
stability  experiment.  Specifically,  we  selected  five  positive

l

D
o
w
n
o
a
d
e
d

f
r
o
m
h

t
t

p
s
:
/
/

i

a
c
a
d
e
m
c
.
o
u
p
.
c
o
m
b
o
n
o
r
m
a

/

f

i

i

l

/

/

t
i
c
s
a
d
v
a
n
c
e
s
/
a
r
t
i
c
e
6
1
/
v
b
a
3
0
6
8
3
8
0
3
4
9
b
y
g
u
e
s
t

f

/

o
n
1
9
M
a
y
2
0
2
6

Figure 2. Impact of temperature on %selected nodes and AUROC. Increasing temperature increases %selected nodes, while performance (AUROC) is
sensitive to %selected nodes, with optimal performance occurring at moderate node selection levels.

8

Abir et al.

l

D
o
w
n
o
a
d
e
d

f
r
o
m
h

t
t

p
s
:
/
/

Figure 3. Robustness to neighborhood noise. AUROC as a function of injected Gaussian noise standard deviation in neighborhood embeddings. The
model’s predictive performance remains stable across a broad range of noise levels, demonstrating strong robustness to neighborhood perturbations.

(true  interacting)  and  five  negative  (non-interacting)  anchor
drug–target  pairs  from  the  BindingDB  test  set  (balanced
warm  start).  For  each  anchor  pair,  we  repeatedly  evaluated
the  predicted  interaction  scores  across  30  different  random
batch  contexts,  each  time  including  the  anchor  pair  among
different sets of other drugs and proteins to construct the sim-
ilarity graphs. The predicted scores for each anchor pair were
collected across all batch contexts. Figure 4 presents boxplots
for  each  anchor  pair,  where  green  boxes  denote  positive
anchors  and  orange  boxes  indicate  negatives.  We  observe
that the prediction scores remain highly stable for both posi-
tive and negative anchor pairs, with  only minimal variation
across batch contexts. The positive anchors consistently yield
high  predicted  probabilities  (mean  �0.85),  while  negative
anchors remain low (mean �0.20), confirming that DCGAT-
DTI’s predictions are robust and not unduly sensitive to the
composition  of  the  batch  neighborhood.  This  demonstrates
the reliability of our approach and indicates that the similar-
ity graph construction process does not introduce significant
prediction variance, further validating the scalability and gen-
eralizability of DCGAT-DTI.

3.7 Ablation study
To  investigate  the  contribution  of  each  component  in  the
DCGAT-DTI  framework,  we  perform  an  ablation  study  by
systematically removing key components and evaluating the
impact  on  performance.  The  results,  presented  in  Table  7,
demonstrate the critical role of dynamic neighborhood selec-
tion,  contrastive  loss,  and  the  DCGAT  module.  We  con-
ducted  this  experiment  on  BindingDB  test  set  (balanced
warm start). The “Without Dynamic neighborhood” variant
selects all nodes from the cross modality, effectively bypass-
ing  the  dynamic  neighborhood  selection  mechanism.  This
leads  to  a  significant  drop  in  both  AUROC  (from  0.943  to
0.893)  and  AUPRC  (from  0.938  to  0.889),  highlighting  the
importance  of  dynamically  selecting  relevant  cross-modal
neighbors to capture meaningful interactions. Removing the
DCGAT  module  entirely  results  in  the  largest  performance

decline,  with  AUROC  and  AUPRC  dropping  to  0.881  and
0.886,  respectively.  This  proves  the  effectiveness  of  the
DCGAT module in modeling intra- and cross-graph interac-
tions to enrich embeddings. Similarly, excluding the contras-
tive  loss  reduces  AUROC  and  AUPRC  to  0.912  and  0.903,
respectively,  demonstrating  its  role  in  aligning  embeddings
for  better  discrimination  of  interacting  and  non-interacting
pairs. All ablation experiments were repeated 10×, and statis-
tical significance was assessed using Wilcoxon rank sum test
(P-value < :05).  Overall,  the  results  validate  the  necessity  of
each  component  in  DCGAT-DTI,  with  dynamic  neighbor-
hood selection, contrastive loss, and the DCGAT module col-
lectively contributing to its robust performance.

3.8 Evaluation on virtual screening benchmarks
In practical drug discovery settings, active binding molecules
are  vastly  outnumbered  by  inactive  (decoy)  molecules.
Standard  global  metrics  such  as  AUROC  and  AUPRC  may
therefore  be  insufficient  for  evaluating  functional  utility,
since  only  the  top-ranked  predictions  are  experimentally
tested in virtual screening campaigns. To address this, we ex-
tend our evaluation to two widely used benchmarks that ex-
plicitly model the active–decoy imbalance: DUD-E (Mysinger
et  al.  2012)  and  LIT-PCBA  Tran-Nguyen  et  al.  (2020).  We
also  include  ColdstartCPI  (Zhao  et  al.  (2025) as  a  baseline
along  with  the  other  baselines  in  this  experiment.  Since
ColdstartCPI  is  trained  on  PDBbind  Wang  et  al.  (2005),
while  DCGAT-DTI  and  the  other  baselines  are  trained  on
DrugBank  for  this  experiment,  we  retrained  ColdstartCPI
with  DrugBank  training  dataset  for  fair  comparison.
Following  common  practice  in  virtual  screening,  we  report
enrichment factor (EF) at 0.5%, 1%, and 2% cutoffs, as well
as  the  BEDROC80:5  score,  which  emphasizes  the  early  re-
trieval of active compounds Zhao et al. (2025). EF quantifies
the fold enrichment of actives among the top-ranked predic-
tions relative to random selection, while BEDROC discounts
later  enrichments  and  assigns  greater  weight  to  ranking
actives  at  the  very  top.  These  metrics  directly  reflect  the

i

a
c
a
d
e
m
c
.
o
u
p
.
c
o
m
b
o
n
o
r
m
a

/

f

i

i

l

/

/

t
i
c
s
a
d
v
a
n
c
e
s
/
a
r
t
i
c
e
6
1
/
v
b
a
3
0
6
8
3
8
0
3
4
9
b
y
g
u
e
s
t

f

/

o
n
1
9
M
a
y
2
0
2
6

DCGAT-DTI

9

Figure. 4 Prediction stability for anchor pairs. Prediction stability of DCGAT-DTI for anchor drug–target pairs across different random batch contexts. Each
boxplot summarizes the predicted class probability of an anchor pair across 30 randomly sampled batch neighborhoods. Green: positive anchors; Orange:
negative anchors.

Table 7. Ablation study. Bold values indicate the best performance.

Model variant

AUROC

AUPRC

DCGAT-DTI (baseline)
Without DCGAT modulea
Without contrastive lossa
Without dynamic neighborhooda

0.943
0.881
0.912
0.893

0.938
0.886
0.903
0.889

a  Statistically significant difference (P-value < :05) from DCGAT-DTI.

practical needs of drug screening, where experimental valida-
tion  is  typically  restricted  to  the  highest-scoring  subset.  As
shown in Table 8, ColdstartCPI obtains slightly better perfor-
mance than DCGAT-DTI on DUD-E. DCGAT-DTI achieves
the  second  best  performance,  outperforming  the  remaining
baselines. On LIT-PCBA, while DTI-LM attains slightly bet-
ter performance, DCGAT-DTI remains competitive.

4 Case study
4.1 Recovering off-target interactions linked
to colitis
To further assess the clinical relevance of our framework, we
conducted a focused case study on off-target interactions un-
derlying  drug-induced  colitis.  Selective  serotonin  reuptake
inhibitors (SSRIs) are widely prescribed antidepressants Reid
and Barbui (2010), and their unintended modulation of sero-
tonin receptors in the gut has been implicated in mucosal in-
flammation  and  microscopic  colitis  (Chojnacki  et  al.  2021,
Rutkowski et al. 2024). This makes SSRIs and their seroto-
nergic off-targets a biologically meaningful test case for our
model. We evaluated SSRI compounds against multiple sero-
tonin  receptor  subtypes.  DCGAT-DTI  assigned  high  proba-
bilities to several established off-target interactions, including
fluoxetine–HTR2C  (0.90),  paroxetine–HTR2A  (0.84),  par-
oxetine–HTR2C (0.80), paroxetine–HTR1D (0.54), and par-
oxetine–HTR1E  (0.51).  At  the  same  time,  it  correctly
de-prioritized non-relevant cases such as fluoxetine–HTR1A
(0.002, ground truth ¼ 0). This prediction profile is pharma-
cologically coherent, concentrating signals on the 5-HT2 fam-
ily
irrelevant  5-HT1A
interactions.  These  results  align  with  established  biological
and clinical evidence: excessive intestinal serotonin signaling
is  known  to  drive  mucosal  inflammation  Chojnacki  et  al.
(2021), and SSRI exposure has been clinically associated with
microscopic colitis Fern�andez-Ba~nares et al. (2007). This case

receptors  while

suppressing

study highlights the potential of DCGAT-DTI to go beyond
on-target binding prediction and to uncover clinically mean-
ingful off-target mechanisms relevant to drug safety.

4.2 Predicting metabolite-protein and metabolite-
transporter interactions
We  further  evaluated  DCGAT-DTI  on  canonical  metabolite-
target  pairs  to  test  its  ability  to  capture  biologically  relevant
interactions  beyond  drug–target  settings.  The  set  included
fumarate-FH,  glucose-SLC2A1,
succinate-SDHA/SUCNR1,
kynurenine-SLC7A5,  AMP/AICAR-adenosine
transporters
(SLC28A3, SLC29A1/2/3), and serotonin-SLC6A4. The model
successfully  recovered  several  metabolite-transporter  interac-
tions,  such  as  serotonin-SLC6A4,  kynurenine-SLC7A5,  and
AMP/AICAR  with  adenosine  transporters,  demonstrating  its
ability  to  capture  meaningful  biological  patterns  in  transport
processes.  By  contrast,  metabolite-enzyme  and  metabolite-
receptor  cases  (e.g.  succinate-SDHA/SUCNR1,  fumarate-FH)
were  not  well  captured.  Overall,  these  results  indicate  that
DCGAT-DTI can extend to metabolite-protein and metabolite-
transporter  interactions,  with  particular  strength  in  trans-
porter cases.

5 Discussion
DCGAT-DTI introduces a novel approach for DTI prediction
by  dynamically  selecting  cross-modal  neighbors  between
drug  and  protein  graphs.  Unlike  existing  models  that  treat
each modality independently, our framework enables mutual
information flow between modalities at each layer through a
CNS  network.  We  show  that  this  DCGAT  significantly
improves  representation  learning  by  integrating  relevant  in-
formation from both modalities, leading to enhanced predic-
tive  performance.  Empirical  results  across  four  benchmark
datasets and three evaluation settings (warm start, cold start
for  drugs,  and  cold  start  for  proteins)  demonstrate  that
DCGAT-DTI  achieves  consistent  improvements  over  state-
of-the-art  baselines.  Our  model  demonstrates  substantial
improvements  in  challenging  settings,  including  unbalanced
data  and  cold  start  scenarios,  where  it  effectively  predicts
interactions  involving  unseen  drugs  or  proteins.  We  further
demonstrate the robustness of DCGAT-DTI through two tar-
geted  analyses.  First,  we  show  that  the  model’s  prediction
performance remains stable across a wide range of perturbed
neighborhood noise levels, highlighting its resilience to input
perturbations. Second, our batch sensitivity analysis confirms

l

D
o
w
n
o
a
d
e
d

f
r
o
m
h

t
t

p
s
:
/
/

i

a
c
a
d
e
m
c
.
o
u
p
.
c
o
m
b
o
n
o
r
m
a

/

f

i

i

l

/

/

t
i
c
s
a
d
v
a
n
c
e
s
/
a
r
t
i
c
e
6
1
/
v
b
a
3
0
6
8
3
8
0
3
4
9
b
y
g
u
e
s
t

f

/

o
n
1
9
M
a
y
2
0
2
6

10

Abir et al.

Table 8. Comparison of DCGAT-DTI and baselines on DUD-E and LIT-PCBA datasets using enrichment-based metrics. Bold values indicate the best
performance.a

Dataset

DUD-E

LIT-PCBA

Model

DCGAT-DTI
ColdstartCPI
DTI-LM
CCL-DTI
TransDTI
CAT-DTI
DCGAT-DTI
ColdstartCPI
DTI-LM
CCL-DTI
TransDTI
CAT-DTI

EF@0.5%

EF@1.0%

EF@2.0%

BEDROC80:5

14.5416
14.9045
12.2176
9.3322
6.9052
4.3245
3.0678
2.5551
3.7550
2.6094
2.0329
1.8986

11.7698
12.0213
10.0990
7.8188
6.0081
4.0820
2.5137
2.1354
2.9790
2.1558
1.7384
1.7125

9.1425
9.2783
8.0025
6.4989
5.0890
3.5669
1.9287
1.5874
2.6860
1.6157
1.5135
1.4872

21.2853
21.7027
18.4784
14.7495
11.5424
8.0284
4.0008
3.1949
4.8925
3.2388
2.8016
2.7001

a  EF indicates fold enrichment of actives at specified cutoffs.

that the predicted score for a drug–target pair is not sensitive
to the specific composition of other drugs and proteins pre-
sent  in  the  batch.  Apart  from  demonstrating  strong  perfor-
mance,  DCGAT-DTI  is  highly  scalable,  requiring  only
approximately 10 minutes to train on a single NVIDIA RTX
A4500 GPU with 24 GB memory.

Despite  its  strong  performance,  DCGAT-DTI  has  certain
limitations.  While  similarity  graphs  are  constructed  using
pretrained  embeddings,  the  model’s  performance  may  vary
based  on  the  quality  of  these  embeddings  or  the  choice  of
similarity  threshold.  Future  work  could  explore  adaptive
graph  construction  strategies,  integrate  structural  and  func-
tional  knowledge  (e.g.  3D  protein  structures  or  binding
affinity),  and  extend  the  model  to  incorporate  temporal  or
multi-context biological data. Our current evaluation in cold-
start settings may still be subject to similarity-based leakage.
In particular, proteins in the training and test sets may share
significant  sequence  homology,  and  drugs  may  share  com-
mon  scaffolds,  which  could  inflate  predictive  performance.
While this issue is common across much of the DTI literature,
a more rigorous evaluation would require removing homolo-
gous proteins or scaffold-similar drugs across splits. We note
this as an important direction for future work.

Overall, DCGAT-DTI sets a new benchmark by demonstrat-
ing how dynamic cross-modal integration can robustly enhance
both representation learning and interaction prediction.

6 Conclusion
In  this  study,  we  introduce  DCGAT-DTI,  a  novel  framework
that  addresses  key  limitations  in  DTI  prediction  by  leveraging
the  DCGAT  module.  Unlike  traditional  approaches,  DCGAT-
DTI dynamically selects cross-modal neighborhoods and jointly
models intra- and cross-graph interactions, effectively capturing
the  complex  dependencies  between  drugs  and  proteins.  Our
dual-objective training strategy, combining BCE loss and super-
vised  contrastive  loss,  further  enhances  the  model’s  ability  to
discriminate  between  true  and  false  interactions.  Through  ex-
tensive evaluations on four widely used benchmark datasets, we
demonstrate  the  state-of-the-art  performance  of  DCGAT-DTI,
highlighting the effectiveness of the DCGAT module in improv-
ing predictive accuracy and generalization. We also show the ef-
fectiveness  of  the  DCGAT  module  in  capturing  meaningful
cross-modal  neighbors  and  enriching  the  representations  of
drugs and proteins. This work establishes a robust foundation
for advancing computational drug discovery and improving the
efficiency of DTI prediction.

Key points
� DCGAT-DTI introduces a novel dynamic cross-graph attention

mechanism that enables drugs and proteins to exchange
information at every layer, moving beyond traditional
approaches that treat each modality independently.

� Our  method  substantially  improves  representation  learning

and predictive accuracy by dynamically selecting relevant
cross-modal neighbors through a CNS network.

� DCGAT-DTI

consistently

outperforms

state-of-the-art

baselines across four benchmark datasets and excels in
challenging scenarios, including cold start and unbalanced
data settings, demonstrating robust generalization to unseen
drugs and proteins.

� DCGAT-DTI  is  resilient  to  neighborhood  noise  and  batch
composition, with prediction performance remaining stable
under noise perturbations.

curation

[equal],  Software

[lead],  Validation

Author contributions
Abrar  Rahman  Abir  (Conceptualization  [equal],  Data  cura-
tion  [equal],  Formal  analysis  [lead],  Methodology  [lead],
Resources
[lead],
Writing—original  draft  [equal]),  Khandakar  Tanvir  Ahmed
(Conceptualization
[equal],
[equal],  Data
Methodology [equal], Supervision [equal], Writing—original
draft [supporting]), Muhtasim Noor Alif (Conceptualization
[supporting],  Formal  analysis  [supporting],  Methodology
[supporting],  Resources  [equal],  Software  [supporting],
[supporting],
Validation
Writing—original
[supporting]),  Wei  Zhang
(Conceptualization  [equal],  Data  curation  [equal],  Funding
acquisition  [lead],  Resources  [equal],  Supervision  [equal],
[equal]),  and  Wencai  Zhang
Writing—original  draft
(Resources  [supporting],  Validation  [supporting],  Writing—
review & editing [equal])

[supporting],  Visualization

draft

Supplementary data
Supplementary  data
Advances online.

are  available  at  Bioinformatics

Conflict of interest
No competing interest is declared.

l

D
o
w
n
o
a
d
e
d

f
r
o
m
h

t
t

p
s
:
/
/

i

a
c
a
d
e
m
c
.
o
u
p
.
c
o
m
b
o
n
o
r
m
a

/

f

i

i

l

/

/

t
i
c
s
a
d
v
a
n
c
e
s
/
a
r
t
i
c
e
6
1
/
v
b
a
3
0
6
8
3
8
0
3
4
9
b
y
g
u
e
s
t

f

/

o
n
1
9
M
a
y
2
0
2
6

DCGAT-DTI

11

Funding
This  work  was  supported  by  grants  from  the  National
Science Foundation (NSF) [NSF-III2246796 to M.N.A., NSF-
III2152030 to K.T.A. and W.Z.].

Data availability
All  preprocessed  datasets  and  source  code  of  DCGAT-DTI
are
https://github.com/compbiolabucf/
DCGAT-DTI.

available

at

References

Abramson J, Adler J, Dunger J et al. Accurate structure prediction of
interactions  with  AlphaFold  3.  Nature  2024;

biomolecular
630:493–500.

Ahmed KT, Park S, Jiang Q et al. Network-based drug sensitivity pre-

diction. BMC Med Genomics 2020;13:193.

Ahmed  KT,  Ansari  MI,  Zhang  W.  DTI-LM:  language  model  powered
drug–target interaction prediction. Bioinformatics 2024;40:btae533.
Baek M, DiMaio F, Anishchenko I et al. Accurate prediction of protein
structures  and  interactions  using  a  three-track  neural  network.
Science 2021;373:871–6.

Ballester  PJ,  Mitchell  JB.  A  machine  learning  approach  to  predicting
protein–ligand binding affinity with applications to molecular dock-
ing. Bioinformatics 2010;26:1169–75.

Brandes  N,  Ofer  D,  Peleg  Y  et  al.  ProteinBERT:  a  universal  deep-
learning  model  of  protein  sequence  and  function.  Bioinformatics
2022;38:2102–10.

Chen L, Tan X, Wang D et al. TransformerCPI: improving compound–
protein interaction prediction by sequence-based deep learning with
experiments.
self-attention  mechanism  and
Bioinformatics 2020;36:4406–14.

reversal

label

Chithrananda S, Grand G, Ramsundar B. ChemBERTa: large-scale self-
supervised  pretraining  for  molecular  property  prediction.  arXiv,
https://doi.org/2010.09885, 2020, preprint: not peer reviewed.
Chojnacki C, Popławski T, Gasiorowska A et al. Serotonin in the path-

ogenesis of lymphocytic colitis. J Clin Med 2021;10:285.

Dehghan  A,  Abbasi  K,  Razzaghi  P  et  al.  CCL-DTI:  contributing  the
interaction  prediction.  BMC

in  drug–target

loss

contrastive
Bioinformatics 2024;25:48.

Elnaggar A, Heinzinger M, Dallago C et al. ProtTrans: toward under-
standing the language of life through self-supervised learning. IEEE
Trans Pattern Anal Mach Intell 2022;44:7112–27.

Faulon J-L, Misra M, Martin S et al. Genome scale enzyme–metabolite
and drug–target interaction predictions using the signature molecu-
lar descriptor. Bioinformatics 2008;24:225–33.

Fern�andez-Ba~nares F, Esteve M, Espin�os JC et al. Drug consumption and
the risk of microscopic colitis. Am J Gastroenterol 2007;102:324–30.
Finkelshtein B, Huang X, Bronstein M et al. Cooperative graph neural
networks.  In:  Proceedings  of  the  41st  International  Conference  on
Machine Learning, arXiv, https://doi.org/2310.01267, 2023, preprint:
not peer reviewed. https://dl.acm.org/doi/10.5555/3692070.3692616
Huang  K,  Xiao  C,  Glass  LM  et  al.  MolTrans:  molecular  interaction
transformer  for  drug–target  interaction  prediction.  Bioinformatics
2021;37:830–6.

Lin Z, Akin H, Rao R et al. Evolutionary-scale prediction of atomic-
level  protein  structure  with  a  language  model.  Science  2023;
379:1123–30.

Liu T, Lin Y, Wen X et al. BindingDB: a web-accessible database of ex-
perimentally  determined  protein–ligand  binding  affinities.  Nucleic
Acids Res 2007;35:D198–201.

Luo Y, Zhao X, Zhou J et al. A network integration approach for drug–
target interaction prediction and computational drug repositioning
from heterogeneous information. Nat Commun 2017;8:573.

Mysinger MM, Carchia M, Irwin JJ et al. Directory of useful decoys, en-
hanced (DUD-E): better ligands and decoys for better benchmark-
ing. J Med Chem 2012;55:6582–94.

Nguyen T, Le H, Quinn TP et al. GraphDTA: predicting drug–target
binding affinity with graph neural networks. Bioinformatics 2021;
37:1140–7.

€Ozt€urk H,  €Ozg€ur A, Ozkirimli E. DeepDTA: deep drug–target binding

affinity prediction. Bioinformatics 2018;34:i821–9.

Rarey M, Kramer B, Lengauer T et al. A fast flexible docking method
using  an  incremental  construction  algorithm.  J  Mol  Biol  1996;
261:470–89.

Reid S, Barbui C. Long term treatment of depression with selective sero-
tonin  reuptake  inhibitors  and  newer  antidepressants.  BMJ  2010;
340:c1468.

Rutkowski K, Udrycka K, Włodarczyk B et al. Microscopic colitis: an
underestimated  disease  of  growing  importance.  J  Clin  Med  2024;
13:5683.

Thafar MA, Olayan RS, Ashoor H et al. DTiGEMSþ: drug–target in-
teraction  prediction  using  graph  embedding,  graph  mining,  and
similarity-based techniques. J Cheminform 2020;12:44–17.

The  UniProt  Consortium.  UniProt:  the  universal  protein  knowledge-
base in 2023. Nucleic Acids Res 2023;51:D523–31. https://doi.org/
10.1093/nar/gkac1052.

Tran-Nguyen V-K, Jacquemard C, Rognan D. LIT-PCBA: an unbiased
data  set  for  machine  learning  and  virtual  screening.  J  Chem  Inf
Model 2020;60:4263–73.

Wang  R,  Fang X,  Lu  Y  et al.  The  PDBbind  database: methodologies

and updates. J Med Chem 2005;48:4111–9.

Wang Y-B, You Z-H, Yang S et al. A deep learning-based method for
drug–target interaction prediction based on long short-term mem-
ory neural network. BMC Med Inform Decis Mak 2020;20:49.
Wei L, Bowen Z, Zhiyong C et al. Exploring local discriminative infor-
mation from evolutionary profiles for cytokine–receptor interaction
prediction. Neurocomputing (Amst) 2016;217:37–45.

Wei L, Long W, Wei L. MDL-CPI: multi-view deep learning model for
interaction  prediction.  Methods  2022;

compound-protein
204:418–27.

Wu H, Liu J, Jiang T et al. AttentionMGT-DTA: a multi-modal drug–
target  affinity  prediction  using  graph  transformer  and  attention
mechanism. Neural Netw 2024;169:623–36.

Yamanishi Y, Araki M, Gutteridge A et al. Prediction of drug–target in-
teraction  networks  from  the  integration  of  chemical  and  genomic
spaces. Bioinformatics 2008;24:i232–40.

Yamanishi Y, Kotera M, Kanehisa M et al. Drug–target interaction pre-
diction from chemical, genomic and pharmacological data in an in-
tegrated framework. Bioinformatics 2010;26:i246–54.

Ye Q, Hsieh C-Y, Yang Z et al. A unified drug–target interaction predic-
tion  framework  based  on  knowledge  graph  and  recommendation
system. Nat Commun 2021;12:6775.

Kalakoti Y, Yadav S, Sundar D. TransDTI: transformer-based language
models  for  estimating  DTIs  and  building  a  drug  recommendation
workflow. ACS Omega 2022;7:2706–17.

Zeng X, Chen W, Lei B. Cat-dti: cross-attention and transformer net-
work  with  domain  adaptation  for  drug–target  interaction  predic-
tion. BMC Bioinformatics 2024;25:141.

Khosla P, Teterwak P, Wang C et al. Supervised contrastive learning.

Adv Neural Inf Process Syst 2020;33:18661–73.

Kim S, Chen J, Cheng T et al. PubChem 2023 update. Nucleic Acids

Res 2023;51:D1373–80.

Knox C,  Wilson  M,  Klinger  CM et  al.  DrugBank 6.0:  the  DrugBank
knowledgebase for 2024. Nucleic Acids Res 2024;52:D1265–75.
Lee I, Keum J, Nam H. DeepConv-DTI: prediction of drug–target inter-
actions  via  deep  learning  with  convolution  on  protein  sequences.
PLoS Comput Biol 2019;15:e1007129.

Zhang W, Chien J, Yong J et al. Network-based machine learning and
graph theory algorithms for precision oncology. NPJ Precis Oncol
2017;1:25.

Zhao Q, Zhao H, Guo L et al. Coldstartcpi: induced-fit theory-guided
dti  predictive  model  with  improved  generalization  performance.
Nat Commun 2025;16:6436.

Zheng S, Li Y, Chen S et al. Predicting drug–protein interaction using
quasi-visual  question  answering  system.  Nat  Mach  Intell  2020;
2:134–40.

l

D
o
w
n
o
a
d
e
d

f
r
o
m
h

t
t

p
s
:
/
/

i

a
c
a
d
e
m
c
.
o
u
p
.
c
o
m
b
o
n
o
r
m
a

/

f

i

i

l

/

/

t
i
c
s
a
d
v
a
n
c
e
s
/
a
r
t
i
c
e
6
1
/
v
b
a
3
0
6
8
3
8
0
3
4
9
b
y
g
u
e
s
t

f

/

o
n
1
9
M
a
y
2
0
2
6

