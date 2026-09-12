#!/usr/bin/env python3
"""Compare repeated validation features before diagnosing model resume drift."""
import argparse
import json
from pathlib import Path
import sys
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]; sys.path.insert(0,str(root/"scripts/training"))
    import train_sc_adaptation as driver
    from pxdesign_train.runner.sc_stream import seed_all
    options = driver.parser().parse_args(["--resume-checkpoint",a.checkpoint,"--output-dir",a.output])
    cfg,recipe = driver.resolve(options)
    output = Path(a.output); output.mkdir(parents=True,exist_ok=True)
    seed_all(cfg.seed)
    components = driver.build_data(cfg,recipe,output)
    report = {}
    for name,panel in components.named_eval_dataloaders.items():
        first = list(panel); second = list(panel)
        differences = []
        for x,y in zip(first,second):
            for section in ("input_feature_dict","label_dict"):
                for key,value in x[section].items():
                    if torch.is_tensor(value) and not torch.equal(value,y[section][key]):
                        other=y[section][key]
                        differences.append(dict(sample=x["sample_id"],section=section,key=key,
                            shape=list(value.shape),max_error=float((value.float()-other.float()).abs().nan_to_num().max())))
        report[name]=differences
        torch.save(first,output/(name.replace("/","_")+".pt"))
    (output/"replay.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__": main()
