from torch.optim import lr_scheduler

def get_scheduler(config, optimizer):
    if config is None:
        scheduler = None
    else:
        name = config['name'].lower()
        if name == 'cosineannealinglr':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
        elif name == 'reducelronplateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                    verbose=1, min_lr=config['min_lr'])
        elif name == 'multisteplr':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
        else:
            assert False
    return scheduler