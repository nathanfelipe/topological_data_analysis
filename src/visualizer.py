# src/visualizer.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from persim import plot_diagrams
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from matplotlib import gridspec


class Visualizer:
    @staticmethod
    def plot_point_cloud(point_cloud):
        if point_cloud.points.shape[1] == 2:
            plt.figure()
            plt.scatter(point_cloud.points[:, 0], point_cloud.points[:, 1])
            plt.axis('equal')
            plt.title("Point Cloud")
            plt.show()
        elif point_cloud.points.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(point_cloud.points[:, 0], point_cloud.points[:, 1], point_cloud.points[:, 2])
            plt.title("Point Cloud")
            plt.show()

    @staticmethod
    def plot_persistence_diagram(diagram):
        plt.figure()
        plot_diagrams(diagram.diagrams)
        plt.title("Persistence Diagram")
        plt.show()

    @staticmethod
    def plot_multiple_persistence_diagrams_video(pd_list, num_pcs):
        """
        Plot multiple persistence diagrams on the same canvas.

        Parameters:
        - pd_list: List of persistence diagrams for each PC.
        - num_pcs: Number of principal components.
        """
        import matplotlib.pyplot as plt
        from persim import plot_diagrams

        num_rows = (num_pcs + 1) // 2
        fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))
        axes = axes.flatten()

        for idx, dgms in enumerate(pd_list):
            ax = axes[idx]
            plot_diagrams(dgms, ax=ax, show=False)
            ax.set_title(f"Persistence Diagrams for PC{idx+1}")

        # Hide any unused subplots
        for i in range(len(pd_list), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_time_series(time_series, extent=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_series.t, y=time_series.x, mode='lines', name='Signal'))
        if extent is not None:
            yr = [min(time_series.x), max(time_series.x)]
            fig.add_shape(type="line", x0=extent[0], y0=yr[0], x1=extent[0], y1=yr[1],
                          line=dict(color="Red", width=2))
            fig.add_shape(type="line", x0=extent[1], y0=yr[0], x1=extent[1], y1=yr[1],
                          line=dict(color="Red", width=2))
        fig.update_layout(title="Original Signal", xaxis_title="Time", yaxis_title="Amplitude")
        fig.show()

    @staticmethod
    def plot_pca_embedding(Y):
        if Y.shape[1] == 2:
            fig = px.scatter(x=Y[:, 0], y=Y[:, 1], title="PCA of Sliding Window Embedding (2D)")
            fig.show()
        elif Y.shape[1] == 3:
            fig = px.scatter_3d(x=Y[:, 0], y=Y[:, 1], z=Y[:, 2],
                                title="Interactive 3D PCA Embedding")
            fig.show()
        else:
            print("Embedding dimension must be 2 or 3 for visualization.")

    @staticmethod
    def plot_pca_eigenvalues(eigenvalues):
        # Assuming eigenvalues is a 1D numpy array
        x = np.arange(1, len(eigenvalues) + 1)
        fig = go.Figure(data=[go.Bar(x=x, y=eigenvalues)])
        fig.update_layout(title="PCA Eigenvalues",
                          xaxis_title="Eigenvalue Number",
                          yaxis_title="Eigenvalue")
        fig.show()

    @staticmethod
    def plot_pca_eigenvalues_video(eigenvalues):
        """
        Plot the eigenvalues (variance explained) from PCA.

        Parameters:
        - eigenvalues: Array of eigenvalues.
        """
        cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, 'o-', label='Eigenvalues')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue (Variance Explained)')
        plt.title('Scree Plot')
        plt.grid(True)
        plt.legend()

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1, len(eigenvalues) + 1), cumulative_variance, 's-', label='Cumulative Variance Explained')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Variance Explained')
        plt.title('Cumulative Variance Explained by PCs')
        plt.grid(True)
        plt.legend()

        plt.show()

    @staticmethod
    def plot_full_visualization(x, extent, Y, eigs):
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 2])

        # Plot the signal
        ax = plt.subplot(gs[0, 0])
        ax.plot(x)
        yr = np.max(x) - np.min(x)
        yr = [np.min(x) - 0.1 * yr, np.max(x) + 0.1 * yr]
        ax.plot([extent[1], extent[1]], yr, 'r')
        ax.plot([extent[0], extent[0]], yr, 'r')
        ax.plot([extent[0], extent[1]], [yr[0]] * 2, 'r')
        ax.plot([extent[0], extent[1]], [yr[1]] * 2, 'r')
        ax.set_title("Original Signal")
        ax.set_xlabel("Sample Number")

        # Create color mapping
        c = plt.get_cmap('jet')
        num_points = Y.shape[0]
        colors = c(np.linspace(0, 1, num_points))

        # Plot the PCA embedding in 3D
        ax2 = plt.subplot(gs[:, 1], projection='3d')
        ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=colors, s=1)
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.set_zlabel("PC3")
        ax2.set_title("PCA of Sliding Window Embedding (3D)")

        # Plot the eigenvalues as bars
        ax3 = plt.subplot(gs[1, 0])
        num_eigs_to_plot = min(len(eigs), 10)
        ax3.bar(np.arange(1, num_eigs_to_plot + 1), eigs[:num_eigs_to_plot])
        ax3.set_xlabel("Eigenvalue Number")
        ax3.set_ylabel("Eigenvalue")
        ax3.set_title("PCA Eigenvalues")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_full_comp5(time_series, extent, PDs, Y, eigs):
        """
        Plot the original signal, persistence diagram, PCA eigenvalues, and 3D PCA embedding.

        Parameters:
        - time_series: An instance of TimeSeries containing the signal.
        - extent: Tuple indicating the extent of the sliding window.
        - PDs: Persistence diagrams computed from the data.
        - Y: PCA-transformed data matrix.
        - eigs: Eigenvalues from PCA.
        """
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2], height_ratios=[2, 2, 1])

        # Plot the original signal
        ax = plt.subplot(gs[0, 1])
        ax.plot(time_series.t, time_series.x)
        ax.set_ylim((1.25 * min(time_series.x), 1.25 * max(time_series.x)))
        ax.set_title("Original Signal")
        ax.set_xlabel("Sample Number")
        yrange = np.max(time_series.x) - np.min(time_series.x)
        yr = [np.min(time_series.x) - 0.1 * yrange, np.max(time_series.x) + 0.1 * yrange]
        ax.plot([extent[1], extent[1]], yr, 'r')
        ax.plot([extent[0], extent[0]], yr, 'r')
        ax.plot([extent[0], extent[1]], [yr[0]] * 2, 'r')
        ax.plot([extent[0], extent[1]], [yr[1]] * 2, 'r')

        # Plot the persistence diagram
        ax2 = plt.subplot(gs[0:2, 0])
        plot_diagrams(PDs, ax=ax2)
        # Find the two largest persistence intervals in H1
        if len(PDs[1]) >= 2:
            persistence_values = PDs[1][:, 1] - PDs[1][:, 0]
            maxind = np.argpartition(persistence_values, -2)[-2:]
            max1 = persistence_values[maxind[0]]
            max2 = persistence_values[maxind[1]]
            ax2.set_title("Persistence Diagram\n Max Pers: %.3g 2nd Pers: %.3g" % (max1, max2))
        else:
            ax2.set_title("Persistence Diagram")

        # Plot the PCA eigenvalues
        ax3 = plt.subplot(gs[2, 0])
        num_eigs_to_plot = min(len(eigs), 10)
        ax3.bar(np.arange(1, num_eigs_to_plot + 1), eigs[:num_eigs_to_plot])
        ax3.set_xlabel("Eigenvalue Number")
        ax3.set_ylabel("Eigenvalue")
        ax3.set_title("PCA Eigenvalues")

        # Plot the 3D PCA embedding
        c = plt.get_cmap('jet')
        num_points = Y.shape[0]
        colors = c(np.linspace(0, 1, num_points))
        ax4 = fig.add_subplot(gs[1:, 1], projection='3d')
        ax4.set_title("PCA of Sliding Window Embedding")
        ax4.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=colors, s=5)
        ax4.set_xlabel("PC1")
        ax4.set_ylabel("PC2")
        ax4.set_zlabel("PC3")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_pca_field_of_coefficients(f1, f2, Y1, Y2, coeff1, coeff2):
        """
        Plot the original signals g1 and g2, along with their 3D PCA embeddings.

        Parameters:
        - f1: Original signal f1.
        - f2: Original signal f2.
        - Y1: PCA-transformed data matrix for f1.
        - Y2: PCA-transformed data matrix for f2.
        - coeff1: Coefficient used in f1.
        - coeff2: Coefficient used in f2.
        """
        c = plt.get_cmap('jet')
        num_points = Y1.shape[0]
        colors = c(np.linspace(0, 1, num_points))[:, :3]

        fig = plt.figure(figsize=(9.5, 6))

        # Plot g1 and its PCA embedding
        ax1 = fig.add_subplot(221)
        ax1.plot(f1)
        ax1.set_title("Original Signal f1")
        ax1.set_xlabel("Sample Index")

        ax2 = fig.add_subplot(222, projection='3d')
        ax2.set_title("f1 = %.2g*cos(t) + %.2g*cos(2t)" % (coeff1, coeff2))
        ax2.scatter(Y1[:, 0], Y1[:, 1], Y1[:, 2], c=colors, s=5)

        # Plot g2 and its PCA embedding
        ax3 = fig.add_subplot(223)
        ax3.plot(f2)
        ax3.set_title("Original Signal f2")
        ax3.set_xlabel("Sample Index")

        ax4 = fig.add_subplot(224, projection='3d')
        ax4.set_title("f2 = %.2g*cos(t) + %.2g*cos(2t)" % (coeff2, coeff1))
        ax4.scatter(Y2[:, 0], Y2[:, 1], Y2[:, 2], c=colors, s=5)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_persistence_diagrams(f1, f2, PDs1_2, PDs1_3, PDs2_2, PDs2_3):
        """
        Plot the signals g1 and g2 along with their persistence diagrams
        computed with different field coefficients.

        Parameters:
        - f1: Original signal f1.
        - f2: Original signal f2.
        - PDs1_2: Persistence diagrams for f1 with coeff=2.
        - PDs1_3: Persistence diagrams for f1 with coeff=3.
        - PDs2_2: Persistence diagrams for f2 with coeff=2.
        - PDs2_3: Persistence diagrams for f2 with coeff=3.
        """
        fig = plt.figure(figsize=(8, 6))

        plt.subplot(231)
        plt.plot(f1)
        plt.title("f1")

        plt.subplot(232)
        plot_diagrams(PDs1_2[1], labels=['H1'])
        plt.title("$f_1$ Persistence Diagram $\mathbb{Z}/2\mathbb{Z}$")

        plt.subplot(233)
        plot_diagrams(PDs1_3[1], labels=['H1'])
        plt.title("$f_1$ Persistence Diagram $\mathbb{Z}/3\mathbb{Z}$")

        plt.subplot(234)
        plt.plot(f2)
        plt.title("f2")

        plt.subplot(235)
        plot_diagrams(PDs2_2[1], labels=['H1'])
        plt.title("$f_2$ Persistence Diagram $\mathbb{Z}/2\mathbb{Z}$")

        plt.subplot(236)
        plot_diagrams(PDs2_3[1], labels=['H1'])
        plt.title("$f_2$ Persistence Diagram $\mathbb{Z}/3\mathbb{Z}$")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_persistence_diagrams_video(dgms, title="Persistence Diagrams"):
        """
        Plot the persistence diagrams.

        Parameters:
        - dgms: List of persistence diagrams.
        - title: Title of the plot.
        """
        plt.figure()
        plot_diagrams(dgms)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_pca_embedding_with_time_series(x, Y, eigs):
        """
        Plot the time series and the PCA embedding.

        Parameters:
        - x: Time series data (1D array).
        - Y: PCA-transformed data (2D or 3D array).
        - eigs: Eigenvalues from PCA.
        """
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        # Plot the time series
        ax1 = plt.subplot(gs[0])
        ax1.plot(x)
        ax1.set_title("Time Series (First Principal Component)")
        ax1.set_xlabel("Time (Frames)")
        ax1.set_ylabel("Amplitude")

        # Plot the PCA embedding
        if Y.shape[1] == 2:
            ax2 = plt.subplot(gs[1])
            ax2.scatter(Y[:, 0], Y[:, 1], s=5, c=np.arange(len(Y)), cmap='viridis')
            ax2.set_title("2D PCA Embedding of Sliding Window")
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
        elif Y.shape[1] >= 3:
            ax2 = plt.subplot(gs[1], projection='3d')
            p = ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=5, c=np.arange(len(Y)), cmap='viridis')
            ax2.set_title("3D PCA Embedding of Sliding Window")
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.set_zlabel("PC3")
            fig.colorbar(p, ax=ax2, label='Time Index')
        else:
            print("Y should have at least 2 dimensions for plotting.")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_multiple_time_series_and_embeddings(time_series_list, embeddings_list, eigs_list):
        """
        Plot multiple time series and their PCA embeddings.

        Parameters:
        - time_series_list: List of time series arrays.
        - embeddings_list: List of PCA-transformed embeddings.
        - eigs_list: List of eigenvalues from PCA.
        """
        num_series = len(time_series_list)
        fig = plt.figure(figsize=(12, 4 * num_series))
        gs = gridspec.GridSpec(num_series, 2, width_ratios=[1, 1])

        for i in range(num_series):
            x = time_series_list[i]
            Y = embeddings_list[i]
            eigs = eigs_list[i]

            # Plot time series
            ax1 = plt.subplot(gs[i, 0])
            ax1.plot(x)
            ax1.set_title(f"Time Series PC{i+1}")
            ax1.set_xlabel("Time (Frames)")
            ax1.set_ylabel("Amplitude")

            # Plot PCA embedding
            if Y.shape[1] == 2:
                ax2 = plt.subplot(gs[i, 1])
                sc = ax2.scatter(Y[:, 0], Y[:, 1], s=5, c=np.arange(len(Y)), cmap='viridis')
                ax2.set_title(f"2D PCA Embedding of Sliding Window (PC{i+1})")
                ax2.set_xlabel("PC1")
                ax2.set_ylabel("PC2")
                fig.colorbar(sc, ax=ax2, label='Time Index')
            elif Y.shape[1] >= 3:
                ax2 = plt.subplot(gs[i, 1], projection='3d')
                sc = ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=5, c=np.arange(len(Y)), cmap='viridis')
                ax2.set_title(f"3D PCA Embedding of Sliding Window (PC{i+1})")
                ax2.set_xlabel("PC1")
                ax2.set_ylabel("PC2")
                ax2.set_zlabel("PC3")
                fig.colorbar(sc, ax=ax2, label='Time Index')
            else:
                print("Y should have at least 2 dimensions for plotting.")

        plt.tight_layout()
        plt.show()