"""Compatibility for trusted local checkpoints in older Ultralytics releases."""
from pathlib import Path
import re


def configure_local_checkpoint_loading(project_root):
    """Keep legacy loading scoped to this project's models/ and runs/ files.

    Ultralytics 8.1 predates PyTorch 2.6's weights_only default. Current
    Ultralytics releases handle that change themselves. No installed package
    files are modified, and explicit weights_only choices remain respected.
    """
    import torch
    import ultralytics
    def version(value):
        return tuple(int(n) for n in re.findall(r'\d+',value.split('+')[0])[:3])
    if version(torch.__version__) < (2,6) or version(ultralytics.__version__) >= (8,3):
        return False
    root=Path(project_root).resolve()
    marker=str(root)
    if getattr(torch.load,'_project_checkpoint_root',None)==marker:
        return True
    original=torch.load
    def load(*args,**kwargs):
        source=args[0] if args else kwargs.get('f')
        if not isinstance(source,(str,Path)):
            source=getattr(source,'name',None)
        if isinstance(source,(str,Path)):
            path=Path(source).resolve()
            if path.is_relative_to(root/'models') or path.is_relative_to(root/'runs'):
                kwargs.setdefault('weights_only',False)
        return original(*args,**kwargs)
    load._project_checkpoint_root=marker
    torch.load=load
    return True
