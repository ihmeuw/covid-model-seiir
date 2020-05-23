import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.patheffects as pe

# TODO: This is my local copy of Directories
# from seiir_model.visualizer.versioner import Directories
# TODO: Comment above and uncomment these  on cluster
from seiir_model_pipeline.core.versioner import Directories
from seiir_model_pipeline.core.versioner import load_regression_settings, load_forecast_settings

ODE_BETA_FIT = "ode_beta_fit"
COEFFICIENTS_FIT = "coefficients_fit"
PARAMETERS_FIT = "parameters_fit"
ODE_COMPONENTS_FORECAST = "ode_forecast"
OUTPUT_DRAWS_CASES = "output_draws_cases"
OUTPUT_DRAWS_DEATHS = "output_draws_deaths"
OUTPUT_DRAWS_REFF = "output_draws_reff"


class Visualizer:

    def __init__(self, directories: Directories,
                 groups: list = None, exclude_groups: list = None,
                 col_group="loc_id", col_date='date'
                 ):
        self.directories = directories
        self.col_group = col_group
        self.col_date = col_date
        self.groups = groups
        if exclude_groups is not None:
            for exclude_group in exclude_groups:
                self.groups.remove(exclude_group)
        self.data = {group: {
            ODE_BETA_FIT: [],
            COEFFICIENTS_FIT: [],
            ODE_COMPONENTS_FORECAST: [],
            OUTPUT_DRAWS_CASES: None,
            OUTPUT_DRAWS_DEATHS: None,
            OUTPUT_DRAWS_REFF: None

        } for group in self.groups}
        self.params_for_draws = []

        # self.metadata = pd.read_csv("../../../data/covid/metadata-inputs/location_metadata_652.csv")
        # TODO: change it for cluster
        # self.metadata = pd.read_csv(directories.get_location_metadata_file(location_set_version_id=652))

        # dictionary of location_id to name
        # TODO: uncomment it for cluster to make Peng's part working
        self.regression_settings = load_regression_settings(directories.regression_version)
        self.forecast_settings = load_forecast_settings(directories.forecast_version)
        self.location_metadata = pd.read_csv(
            self.directories.get_location_metadata_file(
                self.regression_settings.location_set_version_id)
        )
        self.id2loc = self.location_metadata.set_index('location_id')[
            'location_name'].to_dict()

        # read beta regression draws
        for group in groups:
            path_to_regression_draws_for_group = os.path.join(directories.regression_beta_fit_dir, str(group))
            if os.path.isdir(path_to_regression_draws_for_group):
                for filename in os.listdir(directories.regression_beta_fit_dir):
                    if filename.startswith("fit_draw_") and filename.endswith(".csv"):
                        draw_df = pd.read_csv(os.path.join(path_to_regression_draws_for_group, filename))
                        # It's assumed that draw_df contains only the `group` group exclusively
                        self.data[group][ODE_BETA_FIT].append(draw_df)
                    else:
                        continue

        # Params and coefficients are commented out because Peng does not use them curently.

        # # read coefficients draws
        # for filename in os.listdir(directories.regression_coefficient_dir):
        #     if filename.startswith("coefficients_") and filename.endswith(".csv"):
        #         draw_df = pd.read_csv(os.path.join(directories.regression_coefficient_dir, filename))
        #         for group in self.groups:
        #             self.data[group][COEFFICIENTS_FIT].append(draw_df[draw_df['group_id'] == group])
        #     else:
        #         continue
        #
        # # read params draws
        # for filename in os.listdir(directories.regression_parameters_dir):
        #     if filename.startswith("params_draw_") and filename.endswith(".csv"):
        #         draw_df = pd.read_csv(os.path.join(directories.regression_parameters_dir, filename))
        #         self.params_for_draws.append(draw_df)
        #     else:
        #         continue

        # read components forecast
        for group in groups:
            path_to_compartments_draws_for_group = os.path.join(directories.forecast_component_draw_dir, str(group))
            if os.path.isdir(path_to_compartments_draws_for_group):
                for filename in os.listdir(path_to_compartments_draws_for_group):
                    if filename.startswith("draw_") and filename.endswith(".csv"):
                        draw_df = pd.read_csv(os.path.join(path_to_compartments_draws_for_group, filename))
                        self.data[group][ODE_COMPONENTS_FORECAST].append(draw_df)
                    else:
                        continue
            else:
                error_msg = f"ODE Components forecast for the group with {col_group} = {group} is not found"
                print("Error: " + error_msg)
                # raise FileNotFoundError(error_msg)

        #  read final draws
        if os.path.isdir(directories.forecast_output_draw_dir):
            for group in groups:
                self.data[group][OUTPUT_DRAWS_CASES] = pd.read_csv(
                    os.path.join(directories.forecast_output_draw_dir, f"cases_{group}.csv"))
                self.data[group][OUTPUT_DRAWS_DEATHS] = pd.read_csv(
                    os.path.join(directories.forecast_output_draw_dir, f"deaths_{group}.csv"))
                self.data[group][OUTPUT_DRAWS_REFF] = pd.read_csv(
                    os.path.join(directories.forecast_output_draw_dir, f"reff_{group}.csv"))

    def format_x_axis(self, ax, start_date, now_date, end_date,
                      major_tick_interval_days=7, margins_days=5):

        months = mdates.DayLocator(interval=major_tick_interval_days)
        days = mdates.DayLocator()  # Every day
        months_fmt = mdates.DateFormatter('%m/%d')

        # format the ticks
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(months_fmt)
        ax.xaxis.set_minor_locator(days)

        # round to nearest years.
        datemin = np.datetime64(start_date, 'D') - np.timedelta64(margins_days, 'D')
        datemax = np.datetime64(end_date, 'D') + np.timedelta64(margins_days, 'D')
        ax.set_xlim(datemin, datemax)

        # format the coords message box
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')

        if now_date is not None:
            now_date = np.datetime64(now_date, 'D')
            ylims = ax.get_ylim()
            ax.plot([now_date, now_date], ylims, linestyle="dashed", c='black')
            label_level = 0.9*ylims[0] + 0.1*ylims[1]
            ax.text(now_date - np.timedelta64(8, 'D'), label_level, "Past")
            ax.text(now_date + np.timedelta64(2, 'D'), label_level, "Future")

    def plot_ode_compartment(self, group, ax,
                             compartment="I1",
                             linestyle='solid',
                             transparency=0.2,
                             color='orange',
                             draws=None
                             ):

        for i, (past_compartments, future_compartments) in enumerate(zip(self.data[group][ODE_BETA_FIT],
                                                                         self.data[group][ODE_COMPONENTS_FORECAST])):
            if draws is not None:
                if i not in draws:
                    continue

            past_time = pd.to_datetime(past_compartments[self.col_date])
            past_compartment_trajectory = past_compartments[compartment]
            ax.plot(past_time, past_compartment_trajectory,
                    linestyle=linestyle, c=color, alpha=transparency)
            future_time = pd.to_datetime(future_compartments[self.col_date])
            future_compartment_trajectory = future_compartments[compartment]
            ax.plot(future_time, future_compartment_trajectory,
                    linestyle=linestyle, c=color, alpha=transparency)

        # get times
        past_time = pd.to_datetime(self.data[group][ODE_BETA_FIT][0][self.col_date])
        future_time = pd.to_datetime(self.data[group][ODE_COMPONENTS_FORECAST][0][self.col_date])
        start_date = past_time.to_list()[0]
        now_date = past_time.to_list()[-1]
        end_date = future_time.to_list()[-1]

        self.format_x_axis(ax, start_date, now_date, end_date, major_tick_interval_days=14)

    def plot_spline(self):
        num_locs = len(self.data)
        fig, ax = plt.subplots(num_locs, 1,
                               figsize=(8, 4 * num_locs))
        for i, loc_id in enumerate(self.data.keys()):
            for j in range(self.regression_settings.n_draws):
                df = self.data[loc_id][ODE_BETA_FIT].sort_values('date')
                ax[i].scatter(np.arange(df.shape[0]), df['newE_obs'],
                              marker='.', c='#ADD8E6', alpha=0.7)
                ax[i].plot(np.arange(df.shape[0]), df['newE'], c='#008080',
                           alpha=0.5, linewidth=1.0)
            ax[i].set_xticks(np.arange(df.shape[0])[::5])
            ax[i].set_xticklabels(df['date'][::5], rotation='vertical')
            ax[i].set_title()

        plt.savefig(self.directories.regression_diagnostic_dir / 'spline_fit.png',
                    bbox_inches='tight')

    def create_trajectories_plot(self,
                                 group,
                                 output_dir="plots",
                                 compartments=('S', 'E', 'I1', 'I2', 'R', 'beta'),
                                 colors=('blue', 'orange', 'red', 'purple', 'green', 'blue')):
        # TODO: comment 2 and uncomment 1 for cluster
        group_name = self.id2loc[group]
        # group_name = self.metadata[self.metadata['location_id'] == group]['location_name'].to_list()[0]
        fig = plt.figure(figsize=(12, (len(compartments) + 1) * 6))
        grid = plt.GridSpec(len(compartments) + 1, 1, wspace=0.1, hspace=0.4)
        fig.autofmt_xdate()
        for i, compartment in enumerate(compartments):
            ax = fig.add_subplot(grid[i, 0])
            self.plot_ode_compartment(group=group, ax=ax,
                                      compartment=compartment,
                                      linestyle="solid",
                                      transparency=0.1,
                                      color=colors[i],
                                      )
            ax.grid(True)
            # ax.legend(loc="upper left")
            ax.set_title(f"Location {group_name}: {compartment}")
        print(f"Trajectories plot for {group} {group_name} is done")

        plt.savefig(os.path.join(output_dir, f"trajectories_{group_name}.png"))
        plt.close(fig)

    def create_final_draws_plot(self,
                                group,
                                compartments=('Cases', 'Deaths', 'R_effective'),
                                R_effective_in_log=True,
                                output_dir="plots",
                                linestyle="solid",
                                transparency=0.1,
                                color=('orange', 'red', 'blue'),
                                quantiles = (0.05, 0.95)):
        compartment_to_col = {
            'Cases': OUTPUT_DRAWS_CASES,
            'Deaths': OUTPUT_DRAWS_DEATHS,
            'R_effective': OUTPUT_DRAWS_REFF
        }
        # TODO: comment 2 and uncomment 1 for cluster
        group_name = self.id2loc[group]
        #group_name = self.metadata[self.metadata['location_id'] == group]['location_name'].to_list()[0]
        fig = plt.figure(figsize=(12, (3) * 6))
        grid = plt.GridSpec(3, 1, wspace=0.1, hspace=0.4)
        fig.autofmt_xdate()

        for i, compartment in enumerate(compartments):
            compartment_data = self.data[group][compartment_to_col[compartment]]
            ax = fig.add_subplot(grid[i, 0])
            time = pd.to_datetime(compartment_data[self.col_date])
            start_date = time.to_list()[0]
            end_date = time.to_list()[-1]
            now_date = pd.to_datetime(compartment_data[compartment_data['observed'] == 1][self.col_date]).to_list()[-1]
            draw_num = 0
            draws = []
            while f"draw_{draw_num}" in compartment_data.columns:
                draw_name = f"draw_{draw_num}"
                draws.append(compartment_data[draw_name].to_numpy())
                if compartment == "R_effective":
                    if R_effective_in_log is True:
                        ax.semilogy(time, compartment_data[draw_name], linestyle=linestyle, c=color[i], alpha=transparency)
                    else:
                        ax.plot(time, compartment_data[draw_name], linestyle=linestyle, c=color[i], alpha=transparency)
                else:
                    ax.plot(time, compartment_data[draw_name], linestyle=linestyle,
                            c=color[i], alpha=transparency)
                draw_num += 1

            # plot mean and std
            mean_draw = np.nanmean(draws, axis=0)
            low_quantile = np.nanquantile(draws, quantiles[0], axis=0)
            high_quantile = np.nanquantile(draws, quantiles[1], axis=0)
            if compartment == "R_effective":
                ax.plot([start_date, end_date], [1, 1], linestyle='--', c="black")
            if compartment == "R_effective" and R_effective_in_log is True:
                ax.semilogy(time, mean_draw, linestyle="-", c="black", alpha=1, linewidth=1.5)
                ax.semilogy(time, low_quantile, linestyle="-.", c="black", alpha=1, linewidth=1.5)
                ax.semilogy(time, high_quantile, linestyle="-.", c="black", alpha=1, linewidth=1.5)
            else:
                ax.plot(time, mean_draw, linestyle="-", c="black", alpha=1, linewidth=1.5)
                ax.plot(time, low_quantile, linestyle="-.", c="black", alpha=1, linewidth=1.5)
                ax.plot(time, high_quantile, linestyle="-.", c="black", alpha=1, linewidth=1.5)

            self.format_x_axis(ax, start_date=start_date, now_date=now_date, end_date=end_date,
                               major_tick_interval_days=14)
            ax.set_title(f"{group_name}: {compartment}")

        print(f"Final draws plot for {group} {group_name} is done")

        plt.savefig(os.path.join(output_dir, f"final_draws_refflog_{group_name}.png"))
        plt.close(fig)


class PlotBetaCoef:
    def __init__(self,
                 directories: Directories):
        self.directories = directories
        # load settings
        self.settings = load_regression_settings(directories.regression_version)
        self.path_to_location_metadata = self.directories.get_location_metadata_file(
            self.settings.location_set_version_id)
        self.path_to_coef_dir = self.directories.regression_coefficient_dir
        self.path_to_savefig = self.directories.regression_diagnostic_dir

        # load metadata
        self.location_metadata = pd.read_csv(self.path_to_location_metadata)
        self.id2loc = self.location_metadata.set_index('location_id')[
            'location_name'].to_dict()

        # load coef
        df_coef = [
            pd.read_csv(self.directories.get_draw_coefficient_file(i))
            for i in range(self.settings.n_draws)
        ]

        # organize information
        self.covs = np.sort(list(self.settings.covariates.keys()))
        self.covs = np.append(self.covs, 'intercept')
        self.loc_ids = np.sort(list(df_coef[0]['group_id'].unique()))
        self.locs = np.array([
            self.id2loc[loc_id]
            for loc_id in self.loc_ids
        ])
        self.num_locs = len(self.locs)

        # group coef data
        self.coef_data = {}
        for cov in self.covs:
            coef_mat = np.vstack([
                df[cov].values
                for df in df_coef
            ])
            coef_label = self.locs.copy()
            coef_mean = coef_mat.mean(axis=0)
            sort_idx = np.argsort(coef_mean)
            self.coef_data[cov] = (coef_label[sort_idx], coef_mat[:, sort_idx])

    def plot_coef(self):
        for cov in self.covs:
            plt.figure(figsize=(8, self.coef_data[cov][1].shape[1]//4))
            plt.boxplot(self.coef_data[cov][1], vert=False, showfliers=False,
                        boxprops=dict(linewidth=0.5),
                        whiskerprops=dict(linewidth=0.5))
            plt.yticks(ticks=np.arange(self.num_locs) + 1,
                       labels=self.coef_data[cov][0])
            coef_mean = self.coef_data[cov][1].mean()
            plt.vlines(coef_mean, ymin=1, ymax=self.num_locs,
                       linewidth=1.0, linestyle='--', color='#003152')
            plt.grid(b=True)
            plt.box(on=None)
            plt.title(cov)
            plt.savefig(self.path_to_savefig / f'{cov}_boxplot.pdf',
                        bbox_inches='tight')

        # save the coefficient of stats
        for cov in self.covs:
            lower = np.quantile(self.coef_data[cov][1], 0.025, axis=0)
            upper = np.quantile(self.coef_data[cov][1], 0.975, axis=0)
            mean = np.mean(self.coef_data[cov][1], axis=0)
            df = pd.DataFrame({
                'loc': self.locs,
                'loc_id': self.loc_ids,
                'lower': lower,
                'mean': mean,
                'upper': upper,
            })
            df.to_csv(self.path_to_savefig / f'{cov}_coef.csv', index=False)


class PlotBetaResidual:
    def __init__(self,
                 directories: Directories):
        self.directories = directories
        # load settings
        self.settings = load_regression_settings(directories.regression_version)
        self.path_to_location_metadata = self.directories.get_location_metadata_file(
            self.settings.location_set_version_id)
        self.path_to_betas_dir = self.directories.regression_beta_fit_dir
        self.path_to_savefig = self.directories.regression_diagnostic_dir

        # load location metadata
        self.location_metadata = pd.read_csv(self.path_to_location_metadata)
        self.id2loc = self.location_metadata.set_index('location_id')[
            'location_name'].to_dict()

        # load the beta data
        self.df_beta = [
            pd.read_csv(self.directories.get_draw_beta_fit_file(i))[[
                'loc_id',
                'date',
                'days',
                'beta',
                'beta_pred'
            ]].dropna()
            for i in range(self.settings.n_draws)
        ]

        # location information
        self.loc_ids = np.sort(list(self.df_beta[0]['loc_id'].unique()))
        self.locs = np.array([
            self.id2loc[loc_id]
            for loc_id in self.loc_ids
        ])
        self.num_locs = len(self.locs)

    def plot_residual(self):
        for j, loc_id in enumerate(self.loc_ids):
            plt.figure(figsize=(8, 4))
            for i in range(self.settings.n_draws):
                df_sub = self.df_beta[i].loc[self.df_beta[i].loc_id == loc_id]
                days = df_sub['days']
                date = df_sub['date']
                residual = np.log(df_sub['beta']) - np.log(df_sub['beta_pred'])
                plt.plot(days, residual, color='lightseagreen', alpha=0.5)
            plt.xticks(days[::10], date[::10])
            plt.title(f'{loc_id}: beta regression residual')
            plt.savefig(self.path_to_savefig /
                        f'{self.locs[j]}_{loc_id}_residual.png',
                        bbox_inches='tight')
            plt.close('all')


class PlotBetaScaling:
    def __init__(self, directories: Directories):
        self.directories = directories
        # load settings
        self.settings = load_regression_settings(directories.regression_version)
        self.path_to_location_metadata = self.directories.get_location_metadata_file(
            self.settings.location_set_version_id)
        self.path_to_beta_scaling = self.directories.forecast_beta_scaling_dir
        self.path_to_savefig = self.directories.forecast_diagnostic_dir

        # load location metadata
        self.location_metadata = pd.read_csv(self.path_to_location_metadata)
        self.id2loc = self.location_metadata.set_index('location_id')[
            'location_name'].to_dict()

        # load locations
        self.loc_ids = np.array([
            file_name.split('_')[0]
            for file_name in os.listdir(self.path_to_beta_scaling)
        ]).astype(int)
        self.locs = np.array([
            self.id2loc[loc_id]
            for loc_id in self.loc_ids
        ])

        # load data
        self.scales_data = np.hstack([
            pd.read_csv(self.directories.location_beta_scaling_file(loc_id))[
                'beta_scales'
            ].values[:, None]
            for loc_id in self.loc_ids
        ])

    def plot_scales(self):
        scales_mean = self.scales_data.mean(axis=0)
        sort_id = np.argsort(scales_mean)
        plt.figure(figsize=(8, len(self.locs)//4))
        plt.boxplot(self.scales_data[:, sort_id], vert=False, showfliers=False,
                    boxprops=dict(linewidth=0.5),
                    whiskerprops=dict(linewidth=0.5))
        plt.yticks(ticks=np.arange(len(self.locs)) + 1,
                   labels=self.locs[sort_id])
        plt.grid(b=True)
        plt.box(on=None)
        plt.title('beta scalings')
        plt.vlines(1.0, ymin=1, ymax=len(self.locs),
                   linewidth=1.0, linestyle='--', color='#8B0000')
        plt.savefig(self.path_to_savefig/f'beta_scalings_boxplot.pdf',
                    bbox_inches='tight')


if __name__ == "__main__":
    col_date = "date"
    col_group = "loc_id"
    all_groups = [102, 524, 526, 528, 530, 532, 534, 536, 538, 540, 542, 544, 546, 548, 550, 552, 554, 556, 558, 560,
                  562, 564, 566, 568, 570, 572]
    all_groups += [523, 525, 527, 529, 531, 533, 535, 537, 539, 541, 543, 545, 547, 549, 551, 553, 555, 557, 559, 561,
                   563, 565, 567, 569, 571, 573]
    # groups = all_groups
    groups = [526, 528, 530, 534, 572]
    # groups = [524]
    version = "2020_05_03.03"

    directories = Directories(regression_version=version, forecast_version=version)
    visualizer = Visualizer(directories, groups=groups, col_date=col_date, col_group=col_group)

    for group in groups:
       # visualizer.create_trajectories_plot(group=group,
       #                                      # TODO: change when add plotting dirs to the directories object
       #                                      # output_dir = directories.get_trajectories_plot_dir
       #                                      output_dir=".")
        visualizer.create_final_draws_plot(group=group,
                                           # TODO: Same
                                           output_dir=".")
