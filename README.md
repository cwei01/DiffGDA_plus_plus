# DiffGDA++

Core implementation of **DiffGDA++ (PP)** for cross-domain graph learning. This source release contains semantic-consistent unbalanced transport, coupled node–edge flow matching, and generated-source/target hidden-feature MMD alignment. It supports node classification on citation/airport graphs, point-cloud part segmentation, molecular graph classification, and sparse temporal Arxiv graph adaptation.

## Method and code map

1. **Representation and semantic feedback.** A shared encoder learns from source labels. Transport-derived soft target semantics and reliability feed back into encoder training. Original and point-cloud tasks use the shared propagation encoder; molecules use a two-layer GIN.
2. **Semantic-consistent transport.** An iterative unbalanced Sinkhorn solver combines feature and semantic costs. Transport rows provide endpoint sampling probabilities and mass-based training weights.
3. **Joint evolution.** A conditional velocity network jointly evolves latent node states and continuous edge logits. Flow matching trains the velocity field; Heun integration produces transported source features and edge weights.
4. **Classification and alignment.** A separate classifier is trained on transported source data, optionally anchored to original source data. MMD aligns trainable classifier hidden features of generated source and target samples. Target predictions use the target's original graph.

| File | Responsibility |
|---|---|
| `run.py` | One independent training run; no scheduler or hyperparameter search |
| `model/diffgda_pp.py` | Dense PP training, prediction, artifacts and replay |
| `model/pp_training.py` | Encoder feedback training and RBF MMD |
| `model/pp_components.py` | Shared encoder and node-flow components |
| `model/sct.py`, `model/transport.py` | Semantic refinement and unbalanced transport |
| `model/joint_graph.py` | Candidate support, joint/uncoupled velocities, graph head and Heun solver |
| `model/molecular_graph_pp.py` | Molecule-level transport and within-molecule atom evolution |
| `experiment/runtime.py` | Data loading, deterministic execution, evaluation and checkpoint serialization |
| `experiment/pointseg_data.py`, `experiment/pointseg.py` | Point-cloud graph preparation and segmentation evaluation |
| `scripts/molecular_core.py` | Fixed molecular split loading and graph-level evaluation |
| `gdabench/pp.py` | Sparse Arxiv training backend |
| `gdabench/transport.py`, `assignment.py` | Sparse transport and globally consistent assignments |
| `gdabench/flow.py`, `integrate.py` | Sampled flow training and blockwise synchronous integration |
| `gdabench/head_mmd.py` | Sparse classification-head alignment |
| `gdabench/check.py` | Mathematical checks used before sparse training |
| `configs/<family>/` | Complete scientific configurations for each direction |

`encoder_mmd_weight` is the historical parameter name for **classification-head MMD**, not an encoder-stage MMD objective. Arxiv uses `mmd_weight`. Molecular MMD pools atom hidden features first, so each molecule contributes one feature row. Molecular transport matches molecules first, then samples same-category atoms inside each paired molecule; candidate edges never cross molecular boundaries. Continuous latent evolution does not enforce chemical validity of a newly generated molecule.

## Environment

Reference environment: Python 3.10, PyTorch 2.6.0 (CUDA 12.4), PyG 2.8.1, and FAISS CPU 1.9.0.post1. Install PyTorch and its matching PyG extensions first, then run:

```bash
python -m pip install -r requirements.txt
```

Bundled propagation and dataset readers are from [PyGDA](https://github.com/pygda-team/pygda).

## Data layout

Large datasets, trained models and run outputs are excluded. Prepare the data below, or pass `--data-root /path/to/data`. Small fixed split/protocol manifests are included for identity and split inspection.

```text
data/
  Citation/{ACMv9,Citationv1,DBLPv7}/{raw,processed}/
  Airport/{USA,BRAZIL,EUROPE}/{raw,processed}/
  PointSegDA-derived/derived_v1/manifest.json
  PointSegDA-derived/derived_v1/...  # graph files referenced by the manifest
  Mutagenicity/three_domains/{graphs.pt,selection.json,data_audit.json,cache_hash.json}
  GDABench-ArxivTemporal/v1/{protocol.json,A/,B/,C/,evaluation/}
```

Original graph readers use PyGDA's processed/raw layout. Point-cloud loading verifies the fixed derived graph records and checksums; these experiments use 1,024-node graphs and eight part classes. `experiment/pointseg_data.py` contains coordinate normalization, sampling and graph-construction functions. Supplying arbitrary newly sampled clouds does not reproduce the fixed benchmark instances.

Mutagenicity contains 4,337 molecular graphs with 14 atom categories and binary graph labels. Fixed low/medium/high-density domains contain 1,446/1,446/1,445 graphs. Source training uses records marked `source_role=train`, while target adaptation uses the entire target domain with graph labels removed. `selection.json` is the authoritative split. The historical four-domain split is not used. `prepare_molecular.py` rebuilds `graphs.pt` from TU raw files and the included fixed selection rather than inventing a new split. Run `python prepare_molecular.py --raw /path/to/extracted/Mutagenicity --output data/Mutagenicity/three_domains/graphs.pt`. The output must not already exist.

Arxiv domain directories contain `features.npy`, `adjacency.npz`, `pairs.npy`, `ids.npy`, `roots.npy`, and source-label files where required by `gdabench/data.py`; evaluation labels reside separately. Follow `protocol.json` and its hashes. This release expects the prepared temporal benchmark; it does not include a raw-OGB-to-temporal conversion pipeline or a hosted data archive. Data must be supplied separately before benchmark training.

## Run one direction

Run from the repository root. Output directories must be new. The configuration controls scientific hyperparameters; the command supplies family, direction, seed, device and output location.

```bash
# Airport node classification (U=USA, B=BRAZIL)
python run.py --family original --direction U_to_B \
  --config configs/original/U_to_B.json --seed 4 --device cuda:0 \
  --output results/original_U_to_B_seed4

# Point-cloud segmentation (A=Adobe, M=MIT)
python run.py --family point --direction A_to_M \
  --config configs/point/A_to_M.json --seed 2 --device cuda:0 \
  --output results/point_A_to_M_seed2

# Molecular graph classification (H=high density, M=medium density)
python run.py --family molecular --direction H_to_M \
  --config configs/molecular/H_to_M.json --seed 2 --device cuda:0 \
  --selection target_micro --output results/molecular_H_to_M_seed2

# Sparse temporal Arxiv graph adaptation
python run.py --family paper --direction A_to_B \
  --config configs/paper/A_to_B.json --seed 2 --device cuda:0 \
  --output results/paper_A_to_B_seed2
```

Original abbreviations: A=ACM, C=Citation, D=DBLP, U=USA, B=BRAZIL, E=EUROPE. Point-cloud abbreviations: A=Adobe, F=FAUST, M=MIT, S=SCAPE. Molecular abbreviations: L/M/H=low/medium/high density. Arxiv A/B/C are temporal domains; the letter meanings are family-specific.

For original, point and molecular diagnostics, add `--smoke --device cpu` with a fresh output path. This reduces encoder/flow/head budgets and must not be used as a paper score. Arxiv `--smoke` reduces updates but still loads full graph data and builds sparse support; it is not a lightweight toy benchmark.

## Ablation controls

- Disable classification-head MMD with `encoder_mmd_weight=0` (`mmd_weight=0` for Arxiv).
- Dense original/point code supports `flow_state="uncoupled"`: nodes read fixed source edges and edges read initial node states. The molecular and sparse Arxiv core here retains the full method; their separate historical uncoupled patch is not included in this full-method source release.
- Set `joint_balanced=true` to replace UOT by balanced Sinkhorn. This also affects mass-derived weighting and should be described accordingly.

Experiment schedulers, hyperparameter search jobs, private server configuration, datasets, caches, checkpoint archives and working documents are not part of this release.

## Release validation

CPU reduced-budget end-to-end runs passed for original airport graphs, point clouds and molecular graphs, including checkpoint writes. Sparse Arxiv mathematical checks and the FAISS subprocess test passed. Full-budget benchmark training and a full Arxiv run were not repeated for this source packaging task. The source release is prepared locally for review and has not been published to GitHub.
