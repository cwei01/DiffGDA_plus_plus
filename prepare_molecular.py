import argparse,json
from pathlib import Path
from collections import defaultdict
import torch
from torch_geometric.data import Data
FILES={'edges': 'Mutagenicity_A.txt', 'edge_labels': 'Mutagenicity_edge_labels.txt', 'indicator': 'Mutagenicity_graph_indicator.txt', 'graph_labels': 'Mutagenicity_graph_labels.txt', 'node_labels': 'Mutagenicity_node_labels.txt'}
def _ints(path: Path):
    return [int(line.strip()) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]

def _edges(path: Path):
    result = []
    for line in path.read_text(encoding='utf8').splitlines():
        if not line.strip():
            continue
        (left, right) = line.replace(' ', '').split(',')
        result.append((int(left) - 1, int(right) - 1))
    return result

def parse_raw():
    missing = [name for name in FILES.values() if not (RAW / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing extracted TU files: ' + ', '.join(missing))
    indicator = [value - 1 for value in _ints(RAW / FILES['indicator'])]
    node_labels = _ints(RAW / FILES['node_labels'])
    graph_labels = _ints(RAW / FILES['graph_labels'])
    directed = _edges(RAW / FILES['edges'])
    edge_labels = _ints(RAW / FILES['edge_labels'])
    if len(indicator) != len(node_labels):
        raise ValueError('Node label and graph indicator lengths differ')
    if len(directed) != len(edge_labels):
        raise ValueError('Edge and edge-label lengths differ')
    graph_count = max(indicator) + 1
    if graph_count != len(graph_labels):
        raise ValueError('Graph label count does not match graph indicators')
    nodes = defaultdict(list)
    for (global_id, graph_id) in enumerate(indicator):
        nodes[graph_id].append(global_id)
    bonds = defaultdict(dict)
    directed_seen = set()
    for ((left, right), bond_type) in zip(directed, edge_labels):
        if indicator[left] != indicator[right]:
            raise ValueError('A raw bond crosses molecular graph boundaries')
        if left == right:
            raise ValueError('Raw molecular graph contains a self-loop')
        directed_seen.add((left, right, bond_type))
        key = (min(left, right), max(left, right))
        old = bonds[indicator[left]].get(key)
        if old is not None and old != bond_type:
            raise ValueError('Conflicting bond categories for one atom pair')
        bonds[indicator[left]][key] = bond_type
    for (left, right, bond_type) in directed_seen:
        if (right, left, bond_type) not in directed_seen:
            raise ValueError('TU edge list is not bidirectional')
    atom_categories = sorted(set(node_labels))
    raw_classes = sorted(set(graph_labels))
    if len(raw_classes) != 2:
        raise ValueError('Mutagenicity must contain exactly two graph classes')
    atom_to_index = {value: index for (index, value) in enumerate(atom_categories)}
    class_to_index = {value: index for (index, value) in enumerate(raw_classes)}
    records = []
    for graph_id in range(graph_count):
        global_nodes = nodes[graph_id]
        local = {value: index for (index, value) in enumerate(global_nodes)}
        atom_types = [node_labels[value] for value in global_nodes]
        graph_bonds = sorted(((local[a], local[b], label) for ((a, b), label) in bonds[graph_id].items()))
        (n, m) = (len(global_nodes), len(graph_bonds))
        density = 0.0 if n < 2 else 2.0 * m / (n * (n - 1))
        records.append({'graph_id': graph_id + 1, 'atom_types': atom_types, 'bonds': graph_bonds, 'raw_label': graph_labels[graph_id], 'label': class_to_index[graph_labels[graph_id]], 'nodes': n, 'edges': m, 'density': density})
    return (records, atom_categories, raw_classes)
def main():
 global RAW
 p=argparse.ArgumentParser(description="Rebuild molecular PyG graphs using the fixed three-domain selection")
 p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'data/Mutagenicity/three_domains/graphs.pt');a=p.parse_args();RAW=a.raw
 if a.output.exists():raise FileExistsError(a.output)
 selection=json.loads((a.output.parent/'selection.json').read_text());records,categories,classes=parse_raw()
 assert len(records)==4337 and len(categories)==14
 assert {r['graph_id'] for r in records}=={r['graph_id'] for r in selection}
 index={v:i for i,v in enumerate(categories)};graphs=[]
 for r in records:
  types=torch.tensor(r['atom_types'],dtype=torch.long);x=torch.nn.functional.one_hot(torch.tensor([index[v] for v in r['atom_types']]),num_classes=len(categories)).float()
  bonds=r['bonds'];edges=[(i,j) for i,j,k in bonds]+[(j,i) for i,j,k in bonds]
  ei=torch.tensor(edges,dtype=torch.long).t().contiguous() if edges else torch.empty((2,0),dtype=torch.long)
  graph=Data(x=x,edge_index=ei,y=torch.tensor([r['label']]),atom_type=types,graph_id=torch.tensor([r['graph_id']]))
  graphs.append(graph)
 a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(graphs,a.output);print('Saved',len(graphs),'graphs. Validate derived-cache hashes before comparing historical scores.')
if __name__=='__main__':main()
