# Dictionary storing network parameters.
params = {
    'batch_size': 4,   # Batch size.
    'num_epochs': 400,   # Number of epochs to train for.
    'lr': 1e-3,        # Learning rate.
    'beta1': 0.5,
    'beta2': 0.9,
    'cross_valid': 0,
    'save_epoch' : 10,
    'multistep': [150, 250, 350],
    'lr_gamma':0.2,
    'dataset': 'cardiac_dig'}  # MMdataset or cardiac_dig






