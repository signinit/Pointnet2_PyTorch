import os
import numpy as np
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

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
    model = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg))
    all_points = np.loadtxt(cfg.input, delimiter=",")
    indices = np.arange(all_points.shape[0])
    np.random.shuffle(indices)
    np_points = all_points[indices[:4096],:3]
    points = torch.from_numpy(np.array([np_points])).float().cuda()
    model.load_from_checkpoint(cfg.weights)
    model.eval()
    model.to(device)
    results = model(points).detach().cpu()
    print(results[0])
    print(results.size())
    classes = torch.argmax(results, dim=1).numpy()
    np.savetxt("out.txt", np.concatenate([np_points, classes.reshape((4096,1))], axis=1), delimiter=",", fmt="%.6f")

if __name__ == "__main__":
    main()