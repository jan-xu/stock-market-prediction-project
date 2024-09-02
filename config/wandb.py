import wandb

def wandb_config(args):
    print(f"Logging to Wandb: {args.project}/{args.name}")
    wandb.init(
        project=args.project,
        name=args.name,
        config={
            "architecture": args.architecture,
            "dataset": args.dataset,
            "epochs": args.epochs,
            "look_back": args.look_back,
            "pred_horizon": args.pred_horizon,
        }
    )