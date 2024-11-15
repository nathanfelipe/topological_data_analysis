# main.py
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt6.QtCore import Qt
from src.time_series import TimeSeries
from src.visualizer import Visualizer
from matplotlib import gridspec
from ripser import ripser
import numpy as np
import matplotlib.pyplot as plt
from persim import plot_diagrams
import sys


def cosine_signal():
    # Step 1: Setup the signal
    T = 40  # Period in number of samples
    NPeriods = 4  # Number of periods
    N = T * NPeriods  # Total number of samples
    time_series = TimeSeries.generate_cosine_signal(T, NPeriods, N)

    # Parameters for sliding window embedding
    dim = 20
    tau = 1.0
    dt = 0.5

    # Step 2: Perform sliding window embedding
    embedding = time_series.sliding_window_embedding(dim, tau, dt)
    extent = [0, tau * dim]

    # Step 3: Perform PCA
    Y, eigs = time_series.perform_pca_custom(embedding, n_components=2)
    print(f"lambda1 = {eigs[0]}, lambda2 = {eigs[1]}")

    # Step 4: Visualize
    Visualizer.plot_time_series(time_series, extent=extent)
    Visualizer.plot_pca_embedding(Y)


def non_cyclical_time_series():
    # Step 1: Setup the signal with noise
    N = 400  # Number of samples
    x = np.linspace(0, 1, N)
    time_series = TimeSeries(t=np.arange(N), x=x)
    noise_level = 0.05
    time_series.add_noise(noise_level=noise_level)

    # Parameters for sliding window embedding
    dim = 20
    tau = 1.0
    dt = 0.5

    # Step 2: Perform sliding window embedding
    embedding = time_series.sliding_window_embedding(dim, tau, dt)
    extent = [0, tau * dim]

    # Step 3: Perform PCA
    Y, eigs = time_series.perform_pca_custom(embedding, n_components=2)
    print(f"lambda1 = {eigs[0]}, lambda2 = {eigs[1]}")

    # Step 4: Visualize
    Visualizer.plot_time_series(time_series, extent=extent)
    Visualizer.plot_pca_embedding(Y)


def signal_pca_embedding_1():
    # Step 1: Get user inputs
    try:
        dim = int(input("Enter the dimension for sliding window (e.g., 30): "))
        tau = float(input("Enter tau for sliding window (e.g., 1.0): "))
        dt = float(input("Enter dT for sliding window (e.g., 0.5): "))
        second_frequency = float(input("Enter second frequency multiplier (e.g., 3): "))
        noise_level = float(input("Enter noise level (e.g., 0.1): "))
    except ValueError as e:
        print("Invalid input:", e)
        return

    # Step 2: Setup the signal with two frequencies and optional noise
    T1 = 20  # Period of the first sine in number of samples
    T2 = T1 * second_frequency
    NPeriods = 10
    N = int(T2 * NPeriods)
    t = np.arange(N)
    x = np.cos(2 * np.pi * t / T1) + np.cos(2 * np.pi * t / T2)
    x += noise_level * np.random.randn(N)
    time_series = TimeSeries(t=t, x=x)

    # Step 3: Perform sliding window embedding
    try:
        embedding = time_series.sliding_window_embedding(dim, tau, dt)
    except ValueError as e:
        print("Error in sliding window embedding:", e)
        return
    extent = [0, tau * dim]

    # Step 4: Perform PCA with n_components=10
    pca_components = 10  # Compute the first 10 principal components
    pca = time_series.perform_pca_custom_comp(embedding, n_components=pca_components)
    Y = pca['Y']
    eigs = pca['eigenvalues']

    # Step 5: Visualize
    Visualizer.plot_full_visualization(x, extent, Y, eigs)


def signal_pca_embedding_2_persistence():
    # Step 1: Get user inputs
    try:
        dim = int(input("Enter the dimension for sliding window (e.g., 30): "))
        Tau = float(input("Enter tau for sliding window (e.g., 1.0): "))
        noise_amplitude = float(input("Enter noise level (e.g., 0.1): "))
    except ValueError as e:
        print("Invalid input:", e)
        return

    # Step 2: Setup the signal
    T = 40  # The period in number of samples
    NPeriods = 4  # How many periods to go through
    N = T * NPeriods  # The total number of samples
    time_series = TimeSeries.generate_cosine_signal(T, NPeriods, N)

    # Add noise if desired
    noise_amplitude = 0.5  # Adjust as needed
    time_series.add_noise(noise_amplitude)

    # Step 3: Do a sliding window embedding
    # dim = 20  # Dimension of the embedding
    # Tau = 1.0  # Delay parameter
    dT = 0.5  # Time step
    X = time_series.sliding_window_embedding_comp4(dim, Tau, dT)
    extent = Tau * dim

    # Step 4: Do Rips Filtration
    PDs = ripser(X, maxdim=1)['dgms']
    I = PDs[1]

    # Step 5: Perform PCA down to 2D for visualization
    Y, eigs = TimeSeries.perform_pca_custom(X, n_components=2)

    # Step 6: Plot original signal, 2-D projection, and the persistence diagram
    gs = gridspec.GridSpec(2, 2)

    # Plot the original signal
    ax = plt.subplot(gs[0, 0])
    ax.plot(time_series.x)
    ax.set_ylim((2 * min(time_series.x), 2 * max(time_series.x)))
    ax.set_title("Original Signal")
    ax.set_xlabel("Sample Number")
    yr = np.max(time_series.x) - np.min(time_series.x)
    yr = [np.min(time_series.x) - 0.1 * yr, np.max(time_series.x) + 0.1 * yr]
    ax.plot([extent, extent], yr, 'r')
    ax.plot([0, 0], yr, 'r')
    ax.plot([0, extent], [yr[0]] * 2, 'r')
    ax.plot([0, extent], [yr[1]] * 2, 'r')

    # Plot the persistence diagram
    ax2 = plt.subplot(gs[1, 0])
    plot_diagrams(PDs, ax=ax2)
    ax2.set_title("Max Persistence = %.3g" % np.max(I[:, 1] - I[:, 0]))

    # Plot the 2-D PCA projection
    ax3 = plt.subplot(gs[:, 1])
    ax3.scatter(Y[:, 0], Y[:, 1], s=5)
    ax3.axis('equal')
    ax3.set_title("2-D PCA, Eigenvalues: %.3g, %.3g " % (eigs[0], eigs[1]))
    ax3.set_xlabel("PC1")
    ax3.set_ylabel("PC2")

    plt.tight_layout()
    plt.show()


def signal_pca_embedding_3_complete():
    # Step 1: Generate the superposed signal
    T1 = 10  # Period of the first sine in number of samples
    second_freq_multiplier = 1.618  # Multiplier for the period of the second sine
    NPeriods = 5  # Number of periods to go through, relative to the second sinusoid
    noise_amplitude = 0.5  # Amplitude of the added Gaussian noise

    time_series = TimeSeries.generate_superposed_signal(
        T1, second_freq_multiplier, NPeriods, noise_amplitude
    )

    # Step 2: Perform sliding window embedding
    dim = 30  # Embedding dimension
    Tau = 1.0  # Delay parameter
    dT = 0.35  # Time step
    X = time_series.sliding_window_embedding(dim, Tau, dT)
    extent = [0, Tau * dim]

    # Step 3: Compute persistence diagrams
    PDs = ripser(X, maxdim=1)['dgms']

    # Step 4: Perform PCA for visualization
    Y, eigs = TimeSeries.perform_pca_custom(X)

    # Step 5: Plot the results
    Visualizer.plot_full_comp5(time_series, extent, PDs, Y, eigs)


def coefficient_fields():
    # Parameters
    T1 = 95
    T2 = 45
    NPeriods = 4
    coeff1 = 0.3
    coeff2 = 0.4

    # Step 1: Generate the signals
    time_series1, time_series2 = TimeSeries.generate_superposed_signals(T1, T2, NPeriods, coeff1, coeff2)
    f1 = time_series1.x
    f2 = time_series2.x

    # Step 2: Perform sliding window embedding
    dim = 20
    Tau = 5
    dT = 2
    X1 = time_series1.sliding_window_embedding(dim, Tau, dT)
    X2 = time_series2.sliding_window_embedding(dim, Tau, dT)

    # Step 3: Perform PCA
    Y1, eigs1 = TimeSeries.perform_pca_custom(X1)
    Y2, eigs2 = TimeSeries.perform_pca_custom(X2)

    # Step 4: Plot original signals and PCA embeddings
    Visualizer.plot_pca_field_of_coefficients(f1, f2, Y1, Y2, coeff1, coeff2)

    # Step 5: Compute persistence diagrams with different field coefficients
    print("Computing persistence diagrams for f1...")
    PDs1_2 = ripser(X1, maxdim=1, coeff=2)['dgms']
    PDs1_3 = ripser(X1, maxdim=1, coeff=3)['dgms']
    print("Computing persistence diagrams for f2...")
    PDs2_2 = ripser(X2, maxdim=1, coeff=2)['dgms']
    PDs2_3 = ripser(X2, maxdim=1, coeff=3)['dgms']

    # Step 6: Plot the persistence diagrams
    Visualizer.plot_persistence_diagrams(f1, f2, PDs1_2, PDs1_3, PDs2_2, PDs2_3)


def execute_computation7_old_old():
    """
    Step 1: Loads the video using the method from TimeSeries.
    Step 2: Performs PCA to reduce dimensionality and capture significant features.
    Step 3: Computes the time derivative to emphasize dynamic changes.
    Step 4: Performs sliding window embedding to capture temporal correlations.
    Step 5: Normalizes the data to prepare it for TDA.
    Step 6: Computes persistence diagrams using ripser.
    Step 7: Plots the persistence diagrams using the method from Visualizer.
    """
    # Step 1: Load the video
    video_path = 'NormalPeriodicCrop.ogg'  # Adjust the path as necessary
    I, frame_dims = TimeSeries.load_video(video_path)
    print(f"Loaded video with {I.shape[0]} frames of size {frame_dims}")

    # Step 2: Perform PCA on the video data
    print("Performing PCA on video data...")
    X = TimeSeries.generate_pca_video(I)
    print("PCA completed.")

    # Step 3: Compute the time derivative to capture dynamical properties
    derivWin = 10  # Window size for derivative computation
    print("Computing time derivative...")
    X_deriv, validIdx = TimeSeries.get_time_derivative(X, derivWin)
    print("Time derivative computed.")

    # Step 4: Perform sliding window embedding
    dim = 70
    Tau = 0.5
    dT = 1
    print("Performing sliding window embedding...")
    XS = TimeSeries.get_sliding_window_video(X_deriv, dim, Tau, dT)
    print("Sliding window embedding completed.")

    # Step 5: Normalize the data
    XS = XS - np.mean(XS, axis=1, keepdims=True)
    XS = XS / np.linalg.norm(XS, axis=1, keepdims=True)
    print("Data normalization completed.")

    # Step 6: Compute persistence diagrams
    print("Computing persistence diagrams...")
    dgms = ripser(XS, maxdim=2)['dgms']
    print("Persistence diagrams computed.")

    # Step 7: Plot the persistence diagrams
    Visualizer.plot_persistence_diagrams_video(dgms, title="Persistence Diagrams")


def execute_computation7_old():
    # Step 1: Load the video
    video_path = 'data/NormalPeriodicCrop.ogg'  # Adjust the path as necessary
    I, frame_dims = TimeSeries.load_video(video_path)
    print(f"Loaded video with {I.shape[0]} frames of size {frame_dims}")

    # Step 2: Perform PCA on the video data
    print("Performing PCA on video data...")
    # Set the number of components to a reasonable number
    n_components = 50  # Adjust based on your needs and available memory
    X = TimeSeries.generate_pca_video(I, n_components=n_components)
    print("PCA completed.")

    # Step 3: Compute the time derivative
    derivWin = 10  # Window size for derivative computation
    print("Computing time derivative...")
    X_deriv, validIdx = TimeSeries.get_time_derivative(X, derivWin)
    print("Time derivative computed.")

    # Step 4: Perform sliding window embedding
    dim = 70
    Tau = 0.5
    dT = 1
    print("Performing sliding window embedding...")
    XS = TimeSeries.get_sliding_window_video(X_deriv, dim, Tau, dT)
    print("Sliding window embedding completed.")

    # Step 5: Normalize the data
    XS = XS - np.mean(XS, axis=1, keepdims=True)
    XS = XS / np.linalg.norm(XS, axis=1, keepdims=True)
    print("Data normalization completed.")

    # Step 6: Compute persistence diagrams
    print("Computing persistence diagrams...")
    dgms = ripser(XS, maxdim=2)['dgms']
    print("Persistence diagrams computed.")

    # Step 7: Plot the persistence diagrams
    Visualizer.plot_persistence_diagrams_video(dgms, title="Persistence Diagrams")


def video_pca_1():

    """
    Visualizing Temporal Progression:
        In the PCA embedding plots, coloring the points based on their time index helps visualize how the data evolves over time.
        Understanding the Plots:
    Time Series Plot:
        Represents the dominant variation in the video frames over time.
        Peaks and troughs correspond to significant changes in the video content.
    PCA Embedding Plot:
        Shows the structure of the sliding window embedding in reduced dimensions.
        Clusters or patterns may indicate recurring dynamics in the video.

    Potential Adjustments:
    Change n_components in PCA:
        Increasing n_components in generate_pca_video can capture more variance but may increase memory usage.
    Modify derivWin:
        Adjusting the window size for the time derivative may affect the emphasis on dynamics.
    Use Different Time Series:
        Instead of the first principal component, you could extract other features from the video as the time series, such as
        mean pixel intensity.

    :return:
    """
    # Step 1: Load the video
    video_path = 'data/NormalPeriodicCrop.ogg'  # Adjust the path as necessary
    I, frame_dims = TimeSeries.load_video(video_path)
    print(f"Loaded video with {I.shape[0]} frames of size {frame_dims}")

    # Step 2: Perform PCA on the video data
    print("Performing PCA on video data...")
    # Set the number of components to a reasonable number
    n_components = 50  # Adjust based on your needs and available memory
    X = TimeSeries.generate_pca_video(I, n_components=n_components)
    print("PCA completed.")

    # Extract time series from the first principal component
    x = X[:, 0]  # The first principal component over time

    # Step 3: Compute the time derivative
    derivWin = 10  # Window size for derivative computation
    print("Computing time derivative...")
    X_deriv, validIdx = TimeSeries.get_time_derivative(X, derivWin)
    print("Time derivative computed.")

    # Step 4: Perform sliding window embedding
    dim = 30  # Adjust dim based on available memory
    Tau = 1  # Adjust Tau as needed
    dT = 1
    print("Performing sliding window embedding...")
    XS = TimeSeries.get_sliding_window_video(X_deriv, dim, Tau, dT)
    print("Sliding window embedding completed.")

    # Step 5: Normalize the data
    XS = XS - np.mean(XS, axis=1, keepdims=True)
    XS = XS / np.linalg.norm(XS, axis=1, keepdims=True)
    print("Data normalization completed.")

    # Step 6: Perform PCA on the sliding window embedding
    print("Performing PCA on the sliding window embedding...")
    Y, eigs = TimeSeries.perform_pca_custom(XS, n_components=3)
    print("PCA on sliding window embedding completed.")

    # Step 7: Plot the time series and the PCA embedding
    Visualizer.plot_pca_embedding_with_time_series(x, Y, eigs)

    # Step 8: Compute persistence diagrams
    print("Computing persistence diagrams...")
    dgms = ripser(XS, maxdim=2)['dgms']
    print("Persistence diagrams computed.")

    # Step 9: Plot the persistence diagrams
    Visualizer.plot_persistence_diagrams_video(dgms, title="Persistence Diagrams")


def video_pca_heart():
    # Step 1: Load the video
    video_path = 'data/heart_compressed.mp4'  # Adjust the path as necessary
    I, frame_dims = TimeSeries.load_video(video_path)
    print(f"Loaded video with {I.shape[0]} frames of size {frame_dims}")

    # Step 2: Perform PCA on the video data
    print("Performing PCA on video data...")
    n_components = None  # Set to None to compute all components
    X, pca_eigenvalues = TimeSeries.generate_pca_video(I, n_components=n_components, return_eigenvalues=True)
    print("PCA completed.")

    # Plot the PCA eigenvalues (scree plot)
    Visualizer.plot_pca_eigenvalues_video(pca_eigenvalues)

    # Decide on the number of PCs to use based on the scree plot
    num_pcs_to_use = int(input("Enter the number of principal components to use for further analysis: "))
    print(f"Using the first {num_pcs_to_use} principal components.")

    # Extract time series from the selected principal components
    time_series_list = [X[:, i] for i in range(num_pcs_to_use)]

    # Proceed with the rest of the analysis using the selected PCs

    # Initialize lists to store embeddings and eigenvalues
    embeddings_list = []
    eigs_list = []
    dgms_list = []

    # Parameters for sliding window embedding
    dim = 30  # Adjust as needed
    Tau = 1   # Adjust Tau as needed
    dT = 1

    # For each time series, perform sliding window embedding and PCA
    for idx, x in enumerate(time_series_list):
        print(f"Processing PC{idx+1}...")
        # Sliding window embedding
        X_embed = TimeSeries.sliding_window_embedding_1d(x, dim, Tau, dT)
        # Normalize the data
        X_embed = X_embed - np.mean(X_embed, axis=1, keepdims=True)
        X_embed = X_embed / np.linalg.norm(X_embed, axis=1, keepdims=True)
        # Perform PCA on the embedding
        Y, eigs = TimeSeries.perform_pca_custom(X_embed, n_components=3)
        # Store results
        embeddings_list.append(Y)
        eigs_list.append(eigs)
        # Compute persistence diagrams
        print(f"Computing persistence diagrams for PC{idx+1}...")
        dgms = ripser(Y, maxdim=2)['dgms']
        dgms_list.append(dgms)
        print(f"Persistence diagrams for PC{idx+1} computed.")

    # Plot the time series and PCA embeddings
    Visualizer.plot_multiple_time_series_and_embeddings(time_series_list, embeddings_list, eigs_list)

    # Plot all persistence diagrams on the same canvas
    Visualizer.plot_multiple_persistence_diagrams_video(dgms_list, num_pcs_to_use)


class ComputationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manifold Learning: Understanding the shape of data")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create a label
        label = QLabel(" Select a computation to execute: \n"
                    "(hover mouse above options for details)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        # Create buttons for each computation
        computations = [
            ("Cosine Signal", self.run_cosine_signal,
             "Generates a simple cosine signal and 'PCA' it."),
            ("Non-Cyclical TS", self.run_non_cyclical_time_series,
             "Performs superposed cosine signals and computes their sliding window embedding."),
            ("Cyclical TS", self.run_signal_pca_embedding_1,
             "Adds noise to a signal and analyzes its persistence diagrams."),
            ("TS-PCA-PD", self.run_signal_pca_embedding_2_persistence,
             "Compares periodic and chaotic signals using sliding window embedding."),
            ("TS-PCA-Embedding-PD", self.run_signal_pca_embedding_3_complete,
             "Analyzes a signal with multiple frequencies and plots its PCA embedding."),
            ("Field of Coefficients", self.run_coefficient_fields,
        "Analyzes periodic signals with added noise and computes persistence diagrams over a field of coefficients."),
            ("PCA on Video - Throat", self.run_video_pca_1,
             "Processes video data using PCA and computes persistence diagrams."),
            ("PCA on Video - Heart", self.run_video_pca_heart,
             "Processes video data with a user-specified number of principal components."),
        ]

        for name, method, tooltip in computations:
            button = QPushButton(name)
            button.clicked.connect(method)
            button.setToolTip(tooltip)
            layout.addWidget(button)

        # Add an exit button
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close)
        layout.addWidget(exit_button)

        self.setLayout(layout)

    def run_cosine_signal(self):
        self.run_computation(cosine_signal, "Cosine Signal")

    def run_non_cyclical_time_series(self):
        self.run_computation(non_cyclical_time_series, "Non-Cyclical TS")

    def run_signal_pca_embedding_1(self):
        self.run_computation(signal_pca_embedding_1, "Cyclical TS")

    def run_signal_pca_embedding_2_persistence(self):
        self.run_computation(signal_pca_embedding_2_persistence, "TS-PCA-PD")

    def run_signal_pca_embedding_3_complete(self):
        self.run_computation(signal_pca_embedding_3_complete, "TS-PCA-Embedding-PD")

    def run_coefficient_fields(self):
        self.run_computation(coefficient_fields, "Field of Coefficients")

    def run_video_pca_1(self):
        self.run_computation(video_pca_1, "PCA on Video - Throat")

    def run_video_pca_heart(self):
        self.run_computation(video_pca_heart, "PCA on Video - Heart")

    def run_computation(self, computation_func, computation_name):
        try:
            computation_func()
            QMessageBox.information(self, "Success", f"{computation_name} executed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while executing {computation_name}:\n{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ComputationGUI()
    window.show()
    sys.exit(app.exec())

