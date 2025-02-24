import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class TimesGraph:
    def __init__(self):
        # Load the CSV file into a pandas DataFrame
        self.df = pd.read_csv('times-data.csv')

        # Reshape the DataFrame to long format
        # We keep the original image column only for reference during reshaping; it will not be used in the plot.
        time_columns = ["Felzenszwalb Time", "SLIC Time", "Quickshift Time",
                        "OpenCV Centroid Time", "Scikit-image Centroid Time"]
        self.df_long = self.df.melt(id_vars=["Image"], value_vars=time_columns,
                                    var_name="Algorithm", value_name="Time")
        print(self.df_long.head())

    def boxplot(self):
        # Create a boxplot to compare the times of all algorithms
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x="Algorithm", y="Time", data=self.df_long, palette="Set3", hue="Algorithm", legend=False)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.title("Comparison of Algorithm Execution Times")
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
                       inner="quartile", palette="Set3", hue="Algorithm", legend=False)
        plt.xticks(rotation=45)
        plt.title("Algorithm Execution Times - Violin Plot")
        plt.xlabel("Algorithm")
        plt.ylabel("Time (seconds)")
        plt.tight_layout()
        plt.show()

    def strip_plot(self):
        # Strip plot using the 'white' style for a minimalist look
        sns.set(style="white")
        plt.figure(figsize=(10, 6))
        sns.stripplot(x="Algorithm", y="Time", data=self.df_long, jitter=True, palette="muted", hue="Algorithm",
                      legend=False)
        plt.xticks(rotation=45)
        plt.title("Algorithm Times")
        plt.xlabel("Algorithm")
        plt.ylabel("Time (seconds)")
        plt.tight_layout()
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
    times_graph.violinplot()
    times_graph.strip_plot()
    times_graph.point_plot()
    times_graph.lineplot()
    times_graph.histogram()
