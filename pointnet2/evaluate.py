import os
import numpy as np
import sys
import hydra
import omegaconf
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)


@hydra.main("config/config.yaml")
def main(cfg):
    model = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg))
    points = np.loadtxt(sys.argv[1], delimiter=",")[:,:3]
    classes = model(points).numpy()
    np.savetxt("out.txt", np.concatenate([points, classes], axis=1))