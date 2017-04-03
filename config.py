from collections import namedtuple


Configuration = namedtuple('Configuration',
                           'train_batch_size val_batch_size embedding_size '+
                           'max_iteration learning_rate print_loss_every '+
                           'LOG_DIR model_save_filename metadata_filename'
                           )



