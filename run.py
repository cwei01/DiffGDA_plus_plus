"""Single-run entry point for DiffGDA++ (PP)."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/pointseg_vendor')]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--family',choices=['original','point','molecular','paper'],required=True)
 p.add_argument('--direction',required=True,help='e.g. U_to_B, A_to_M, H_to_M, A_to_B')
 p.add_argument('--config',type=Path,required=True);p.add_argument('--seed',type=int,default=2)
 p.add_argument('--device',default='cuda:0');p.add_argument('--output',type=Path,required=True)
 p.add_argument('--data-root',type=Path,default=ROOT/'data')
 p.add_argument('--selection',choices=['target_micro','source_holdout_macro'],default='target_micro')
 p.add_argument('--smoke',action='store_true',help='Reduced-budget diagnostic, not a paper result')
 a=p.parse_args();cfg=json.loads(a.config.read_text());out=a.output.resolve()
 if out.exists():raise FileExistsError('Choose a fresh output directory')
 out.mkdir(parents=True)
 if a.smoke:
  cfg.update(encoder_epochs=3,flow_epochs=2,head_epochs=3)
 import torch
 from experiment.runtime import configure_determinism,seed_everything,Timings,Evaluator
 configure_determinism('strict');seed_everything(a.seed)
 from experiment.cli import parse_args
 if a.family=='paper':
  from gdabench import data,pp
  data.DATA=data.BASE=a.data_root/'GDABench-ArxivTemporal/v1'
  # Imported path constants are rebound explicitly for relocatable data.
  pp.DATA=data.DATA;pp.BASE=data.BASE;pp.ROOT=out
  from gdabench.check import validate_math
  validate_math()
  (out/'validation.json').write_text(json.dumps(dict(status='passed',code=pp.implementation_hashes()),indent=2))
  cfg.update(seed=a.seed,task=a.direction,candidate='RUN',device=a.device,version='reviewer-pp-core')
  if a.smoke:
   for k in ('encoder_updates','flow_updates','head_updates'):cfg[k]=3
  pp.configuration=lambda candidate,task:dict(cfg)
  (out/'spec.json').write_text(json.dumps(dict(config=cfg,selection='fixed_budget_final',smoke=a.smoke),indent=2))
  state=pp.train('RUN',a.direction,resume=False)
  print(json.dumps(state,default=str));return
 flags=['--method','diffgda_pp','--seed',str(a.seed),'--device',a.device,'--deterministic','strict','--artifact-policy','full']
 defaults=parse_args(flags)
 if a.family=='molecular':
  from scripts import molecular_core as core
  core.BASE=a.data_root/'Mutagenicity/three_domains'
  source,valid,target,gold=core.split_graphs(a.direction.replace('_to_','_'))
  if a.selection=='target_micro':
   graphs=torch.load(core.BASE/'graphs.pt',map_location='cpu',weights_only=False)
   labels={int(g.graph_id):int(g.y) for g in graphs}
   observer=core.HoldoutObserver(target,[labels[int(g.graph_id)] for g in target],a.device,domain='target',selection='target_micro')
  else:observer=core.HoldoutObserver(valid,gold,a.device)
  from types import SimpleNamespace
  args=SimpleNamespace(**(vars(defaults)|dict(encoder='gin2',batch_size=64)|cfg))
  from model.molecular_graph_pp import fit_molecular_pp
  art=fit_molecular_pp(source,target,args,a.device,on_head_epoch=observer)
  torch.save(dict(config=cfg,state=observer.best_state),out/'best.pt')
  torch.save(dict(config=cfg,state=art.predictor.state_dict()),out/'final.pt')
  result=dict(best=observer.best,final=observer.final,history=art.histories,selection=a.selection,smoke=a.smoke)
 else:
  args=parse_args(flags+[x for k,v in cfg.items() for x in (['--'+k.replace('_','-')] if v is True else [] if v is False else ['--'+k.replace('_','-'),str(v)])])
  timers=Timings(a.device)
  if a.family=='original':
   from experiment.runtime import load_graph,prepare_pair
   mapping={'A':'ACMv9','C':'Citationv1','D':'DBLPv7','U':'USA','B':'BRAZIL','E':'EUROPE'}
   s,t=a.direction.split('_to_');source,target,labels,classes=prepare_pair(load_graph(a.data_root,mapping[s]),load_graph(a.data_root,mapping[t]));evaluator=Evaluator(labels,timers)
  else:
   from experiment.pointseg import prepare_graphs,SegmentationEvaluator
   mapping={'A':'adobe_0','F':'faust_0','M':'mit_0','S':'scape_0'};s,t=a.direction.split('_to_')
   source,target,labels,_,_=prepare_graphs(a.data_root/'PointSegDA-derived/derived_v1',mapping[s],mapping[t]);classes=list(range(8));evaluator=SegmentationEvaluator(labels,timers)
  from model.diffgda_pp import DiffGDAPPMethod
  method=DiffGDAPPMethod(args,len(classes));diag=method.fit(source.to(a.device),target.to(a.device),evaluator)
  evaluator.save(out/'checkpoints',dict(config=cfg,seed=a.seed));method.save_artifacts(out/'checkpoints')
  result=dict(best=evaluator.best,final=evaluator.final,diagnostics=diag,selection='target_label_assisted_best_micro',smoke=a.smoke)
 (out/'spec.json').write_text(json.dumps(dict(config=cfg,seed=a.seed,family=a.family,direction=a.direction,selection=result['selection']),indent=2))
 (out/'result.json').write_text(json.dumps(result,indent=2,default=str));print(json.dumps(result['best']))
if __name__=='__main__':main()
