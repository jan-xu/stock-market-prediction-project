import wandb


def wandb_config(args):
    print(f"Logging to Wandb: {args.project}/{args.name}\n")
    wandb.init(
        project=args.project,
        name=args.name,
        config={
            "device": args.device.type,
            "architecture": args.architecture,
            "ticker": args.ticker,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "look_back": args.look_back,
            "pred_horizon": args.pred_horizon,
            "hidden_width": args.hidden_width,
            "dropout": args.dropout,
            "val_size": args.val_size,
            "recurrent_pred_horizon": args.recurrent_pred_horizon,
        },
    )
