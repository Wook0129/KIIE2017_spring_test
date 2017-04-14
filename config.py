from collections import namedtuple


Configuration = namedtuple('Configuration',
                           'train_batch_size val_batch_size embedding_size '+
                           'max_iteration print_loss_every '+
                           'LOG_DIR model_save_filename metadata_filename '+
                           'corruption_ratio_train corruption_ratio_validation'
                           )



