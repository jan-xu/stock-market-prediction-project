def overwrite_fast_toy_args(args):
    args.device = "cpu"
    args.architecture = "LSTM"
    args.epochs = 10
    args.batch_size = 64
    args.look_back = 5
    args.pred_horizon = 1
    args.hidden_width = 128
    args.dropout = 0.0
    args.val_size = 1
    args.train_logs = 2
    args.val_logs = 2
    args.recurrent_pred_horizon = False
    args.eda = False
    args.wandb = False
    args.ignore_timestamp = False
    args.TOY = True
    return args
