from .hgraph.hgnn import HierVAE, HierVGNN, HierCondVGNN
from .vjtnn.diff_vae import DiffVAE
from .jtvae.jtnn_vae import JTNNVAE

class ModelFactory:
    @staticmethod
    def get_model(params):
        if params.model == 'hiervae':
            return HierVAE(params)
        if params.model == 'hiervgnn':
            if params.conditional:
                return HierCondVGNN(params)
            else:
                return HierVGNN(params)
        if params.model == 'vjtnn':
            return DiffVAE(params)
        if params.model == 'jtvae':
            return JTNNVAE(params)
