from ts_analysis import EDA

def run_eda(df, run_folder, label, value_col, date_col=None, volume_col=None, add_ma=False):

    eda = EDA(data=df, label=label, value_col=value_col)
    if add_ma:
        eda.add_moving_average(window=[7, 15, 30], column=eda.data[eda.value_col])
        eda.add_moving_average(window=[7, 15, 30], column=eda.data[eda.return_col])

    logged_plots = []

    eda.plot_price(plot_ma=add_ma, save_fig_name=f"{run_folder}/figures/png/{eda.label}_price_plot", save_fmts=["png"])
    eda.plot_price(plot_ma=add_ma, save_fig_name=f"{run_folder}/figures/html/{eda.label}_price_plot", save_fmts=["html"])
    eda.plot_return(plot_ma=add_ma, save_fig_name=f"{run_folder}/figures/png/{eda.label}_return_plot", save_fmts=["png"])
    eda.plot_return(plot_ma=add_ma, save_fig_name=f"{run_folder}/figures/html/{eda.label}_return_plot", save_fmts=["html"])

    logged_plots.extend([
        f"{run_folder}/figures/png/{eda.label}_price_plot.png",
        f"{run_folder}/figures/html/{eda.label}_price_plot.html",
        f"{run_folder}/figures/png/{eda.label}_return_plot.png",
        f"{run_folder}/figures/html/{eda.label}_return_plot.html"
    ])

    if eda.volume_col is not None:
        eda.plot_volume(save_fig_name=f"{run_folder}/figures/png/{eda.label}_volume_plot", save_fmts=["png"])
        eda.plot_volume(save_fig_name=f"{run_folder}/figures/html/{eda.label}_volume_plot", save_fmts=["html"])
        logged_plots.extend([
            f"{run_folder}/figures/png/{eda.label}_volume_plot.png",
            f"{run_folder}/figures/html/{eda.label}_volume_plot.html"
        ])

    if eda.date_col is not None:
        eda.stl_decomposition(save_fig_name=f"{run_folder}/figures/png/{eda.label}_stl_decomposition", save_fmts=["png"])
        eda.stl_decomposition(save_fig_name=f"{run_folder}/figures/html/{eda.label}_stl_decomposition", save_fmts=["html"])
        logged_plots.extend([
            f"{run_folder}/figures/png/{eda.label}_stl_decomposition.png",
            f"{run_folder}/figures/html/{eda.label}_stl_decomposition.html"
        ])

    print(f"Logged plots: \n-> {'\n-> '.join(logged_plots)}\n")
    print("Summary of synthetic data:")
    print(eda.get_summary(), "\n")
    if eda.date_col is None:
        print(f"Index range: {eda.get_index_range()}\n")
    else:
        print(f"Date range: {eda.get_date_range()}\n")

    return logged_plots
