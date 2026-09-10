"""Exact previously runtime-tested logger/buffer helpers; no old checkpoint transfer."""
import torch

def prepare_algorithm_buffers_for_load(algorithm):
    """Exact reviewed inference-buffer repair from isaaclab/train_mkii_fourbar.py.

    Copied locally to avoid importing a different physical-task entrypoint, which
    also changes import paths. Only inference buffers are replaced, preserving values.
    """
    replaced=[]
    for model_name in ('actor','critic','rnd'):
        model=getattr(algorithm,model_name,None)
        if model is None:continue
        for module_name,module in model.named_modules():
            for name,tensor in tuple(module.named_buffers(recurse=False)):
                if tensor.is_inference():
                    with torch.inference_mode(False):setattr(module,name,tensor.clone())
                    replaced.append('.'.join(filter(None,(model_name,module_name,name))))
    return replaced

def prepare_local_logger_for_early_checkpoint(runner):
    """RSL 5.0.1 defers writer creation until learn, but save_model reads it.

    Initialize only the absent local-writer sentinel. Calling the full writer
    initializer here would create it twice when learn starts. No logging config,
    learned tensor, normalizer or optimizer state changes.
    """
    logger=runner.logger
    if not hasattr(logger,'writer'):
        if logger.cfg.get('logger')!='tensorboard' or logger.log_dir is None:
            raise ValueError('Early checkpoint requires the configured local TensorBoard logger')
        logger.writer=None
