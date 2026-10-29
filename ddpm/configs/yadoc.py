import yaml

class DotConfig:
    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k):
        if k in self._cfg:
            v = self._cfg[k]
            if isinstance(v, dict):
                return DotConfig(v)
            # IMPROVEMENT 1: handle lists containing dicts
            if isinstance(v, list):
                return [DotConfig(i) if isinstance(i, dict) else i for i in v]
            return v
        else:
            raise AttributeError(f"'DotConfig' object has no attribute '{k}'")

    # ==========================================
    # FOR DEBUGGING AND LOGGING
    # ==========================================

    def __repr__(self):
        # IMPROVEMENT 2: makes print(config) show the actual dictionary nicely
        return repr(self._cfg)

    def to_dict(self):
        # returns pure python dict (very useful for WandB or TensorBoard logging)
        return self._cfg

    # IMPROVEMENT 3: add standard dict methods to make it iterable
    def items(self):
        return self._cfg.items()
        
    def keys(self):
        return self._cfg.keys()
        
    def values(self):
        return self._cfg.values()

    def __contains__(self, key):
        # allows using "if 'batch_size' in config.train_params:"
        return key in self._cfg

def parameters(fname):
    with open(fname, 'r') as f:
        doc = yaml.load(f, Loader=yaml.FullLoader)
        return DotConfig(doc)