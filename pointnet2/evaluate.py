import os
import math
import numpy as np
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pointnet2.models import PointNet2SemSegSSG
from pointnet2.models import PointNet2ClassificationSSG

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
    device = torch.device("cuda")
    all_points = np.loadtxt(cfg.input, delimiter=",")[:,:3]
    batches = math.ceil(all_points.shape[0] / 4096)
    np_points = np.resize(all_points, (batches, 4096, 3))
    points = torch.from_numpy(np_points).float().cuda()
    
    print(cfg.task_model)

    model = cfg.task_model['class'].load_from_checkpoint(cfg.weights)
    model.eval()
    model.to(device)
    results = model(points).detach().cpu()
    classes = torch.argmax(results, dim=1).numpy()
    np.savetxt("out.txt", np.concatenate([all_points, classes.reshape((-1,1))[:len(all_points)]], axis=1), delimiter=",", fmt="%.6f")

if __name__ == "__main__":
    main()