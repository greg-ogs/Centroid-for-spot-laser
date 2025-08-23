import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from typing import Optional


class TimesGraph:
    def __init__(self, r_melt=False, csv_input_path=None):
        if r_melt:
            self.df_long = self.melt(csv_input_path)
        else:
            # Load the CSV file into a pandas DataFrame
            self.df = pd.read_csv('times-R-5600X-peak-intensity-detection.csv')

            # Reshape the DataFrame to long format
            # We keep the original image column only for reference during reshaping; it will not be used in the plot.
            time_columns = ["Felzenszwalb Time", "SLIC Time", "Quickshift Time",
                            "FBM Time", "CCL Time"]
            self.df_long = self.df.melt(id_vars=["Image"], value_vars=time_columns,
                                        var_name="Algorithm", value_name="Time")
            print(self.df_long.head())

    @staticmethod
    def melt(csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Melt the dataset in 'times-comparative-peak-intensity-detection.csv' into long format.

        Parameters:
            csv_path: Optional explicit path to the CSV. If None, the file is resolved
                      relative to the project root (one level up from this script).

        Returns:
            pandas.DataFrame: Long-format DataFrame with columns [<id_vars>..., 'Algorithm', 'Time'].
        """
        # Resolve default path to the CSV at the project root if not provided
        if csv_path is None:
            base_dir = os.path.dirname(__file__)
            csv_path = os.path.abspath(os.path.join(base_dir, "..", "times-comparative-peak-intensity-detection.csv"))

        # Load the CSV and identify time columns (those ending with ' Time')
        df_comp = pd.read_csv(csv_path)
        time_columns = [col for col in df_comp.columns if isinstance(col, str) and col.endswith(" Time")]
        if not time_columns:
            raise ValueError("No time columns found to melt in the provided CSV.")

        id_vars = [col for col in df_comp.columns if col not in time_columns]

        # Melt into long format
        df_comp_long = df_comp.melt(id_vars=id_vars, value_vars=time_columns,
                                         var_name="Algorithm", value_name="Time")

        # Quick preview for verification during interactive runs
        print(df_comp_long.head())
        df_comp_long.to_csv("long-times-comparative-peak-intensity-detection.csv", index=False)
        return df_comp_long

    def boxplot(self):
        # Create a boxplot to compare the times of all algorithms
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 8))
        ax = sns.boxplot(x="Algorithm", y="Time", data=self.df_long, palette="plasma", hue="Algorithm", legend=False)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        # plt.title("Comparison of Algorithm Execution Times")
        plt.ylabel("Time (seconds)", fontsize=20)
        plt.xlabel("Algorithm")
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=20)
        plt.tight_layout()
        plt.savefig("times-boxplot.png")
        # Display the plot
        plt.show()

    def boxplot_long(self):
        # Create a boxplot to compare the times of all algorithms
        sns.set(style="whitegrid")
        # plt.figure(figsize=(10, 6))
        ax = sns.catplot(kind="box", x="Algorithm", y="Time", data=self.df_long, palette="plasma", hue="Algorithm",
                         col="CPU", legend=False, sharey=True, sharex=True, col_wrap=2, height=6, aspect=1.5,
                         margin_titles=True, legend_out=False)

        # # Rotate x-axis labels for better readability
        # plt.xticks(rotation=45)
        # plt.title("Comparison of Algorithm Execution Times")
        plt.ylabel("Time (seconds)")
        plt.xlabel("Algorithm")
        plt.tight_layout()

        # Display the plot
        plt.show()

    def violinplot(self):
        # New violin plot method using a different style and seaborn functions
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Algorithm", y="Time", data=self.df_long,
                       inner="quartile", palette="plasma", hue="Algorithm", legend=False)
        plt.xticks(rotation=45)
        plt.title("Algorithm Execution Times - Violin Plot")
        plt.xlabel("Algorithm")
        plt.ylabel("Time (seconds)")
        plt.tight_layout()
        plt.show()

    def strip_plot(self):
        # Strip plot using the 'white' style for a minimalist look
        sns.set(style="white")
        # plt.figure(figsize=(10, 6))
        plt.rcParams['figure.figsize'] = 14, 14
        plt.rcParams.update({'font.size': 45})
        sns.stripplot(x="Algorithm", y="Time", data=self.df_long, jitter=True, palette="colorblind", hue="Algorithm",
                      legend=False)
        plt.xticks(rotation=45)
        # plt.title("Algorithm Times")
        plt.xlabel("Algorithm", fontsize=25)
        plt.ylabel("Time (seconds)", fontsize=25)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        # plt.legend(fontsize=30, loc="upper left")
        plt.tight_layout()
        plt.savefig("times-plot.png")
        plt.show()

    def point_plot(self):
        # Point plot using 'dark' style to display means with confidence intervals
        sns.set(style="dark")
        plt.figure(figsize=(10, 6))
        sns.pointplot(x="Algorithm", y="Time", data=self.df_long, palette="bright", capsize=0.1, hue="Algorithm",
                      legend=False)
        plt.xticks(rotation=45)
        plt.title("Algorithm Execution Times - Point Plot")
        plt.xlabel("Algorithm")
        plt.ylabel("Time (seconds)")
        plt.tight_layout()
        plt.show()

    def lineplot(self):
        # Lineplot to visualize execution times across images for each algorithm
        sns.set(style="whitegrid")
        plt.figure(figsize=(100, 60))
        # Plot a line for each algorithm using the 'hue' parameter.
        # This assumes that the 'Image' column has an order that makes sense for a lineplot.
        sns.lineplot(x="Image", y="Time", hue="Algorithm", marker=".", data=self.df_long, legend=False)
        plt.xticks(rotation=45)
        plt.title("Algorithm Execution Times per Image")
        plt.xlabel("Image")
        plt.ylabel("Time (seconds)")
        plt.tight_layout()
        plt.show()

    def histogram(self):
        # Histogram to visualize the distribution of execution times for each algorithm
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        # Using stat="density" to normalize the histogram if desired.
        sns.histplot(data=self.df_long, x="Time", hue="Algorithm", element="step",
                     stat="density", common_norm=False, palette="pastel", bins=20, legend=False)
        plt.title("Distribution of Algorithm Execution Times")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    times_graph = TimesGraph()
    times_graph.boxplot()
    # times_graph.violinplot()
    # times_graph.strip_plot()
    # times_graph.point_plot()
    # times_graph.lineplot()
    # times_graph.histogram()
    # TimesGraph.melt()
