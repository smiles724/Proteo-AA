#!/usr/bin/env python3
"""Compare frozen feature/SC inputs and predictions across reloads on fixed data."""
import argparse
import gc
import json
from pathlib import Path
import torch
from pxdesign_train.checkpoints import evaluation_model
from pxdesign_train.runner.sc_stream import seed_all


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--batch",required=True);p.add_argument("--output",required=True);p.add_argument("--deterministic",action="store_true")
    a=p.parse_args(); batch=torch.load(a.batch,map_location="cpu",weights_only=False)[0]
    if a.deterministic: torch.use_deterministic_algorithms(True)
    def move(value):
        if torch.is_tensor(value): return value.cuda()
        if isinstance(value,dict): return {key:move(x) for key,x in value.items()}
        return value
    snapshots=[]
    for instance in range(2):
        model=evaluation_model(a.checkpoint,device="cuda")
        # This diagnostic explicitly compares numeric modes, overriding the
        # checkpoint protocol's deterministic default when the flag is absent.
        torch.use_deterministic_algorithms(a.deterministic)
        for repeat in range(2):
            inputs={}
            def capture(module,args,kwargs):
                for key,value in list(enumerate(args))+list(kwargs.items()):
                    if torch.is_tensor(value): inputs[str(key)]=value.detach().float().cpu().clone()
            hook=model.sidechain_module.register_forward_pre_hook(capture,with_kwargs=True)
            seed_all(1000045)
            tensor=move(batch)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16,cache_enabled=False):
                out=model(input_feature_dict=tensor["input_feature_dict"],label_dict=tensor["label_dict"],mode="train")
            inputs["prediction"]=out["sc_pred_global"].float().cpu()
            inputs["a_token"]=model._a_token_cache.float().cpu()
            inputs["loss"]=out["sc_gt_mse"].float().cpu()
            snapshots.append(inputs);hook.remove();del out,tensor
        del model;gc.collect();torch.cuda.empty_cache()
    report=[]
    for index,snapshot in enumerate(snapshots[1:],1):
        report.append(dict(index=index,differences={key:float((value-snapshot[key]).abs().max()) for key,value in snapshots[0].items() if not torch.equal(value,snapshot[key])}))
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__": main()
