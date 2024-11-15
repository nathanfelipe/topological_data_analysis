# src/time_series.py

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.decomposition import PCA
import cv2
from scipy.interpolate import interp1d


class TimeSeries:
    def __init__(self, t=None, x=None):
        self.t = t
        self.x = x

    def add_noise(self, noise_level):
        noise = noise_level * np.random.randn(len(self.x))
        self.x += noise

    @staticmethod
    def generate_superposed_signal(T1, second_freq_multiplier, NPeriods, noise_amplitude):
        """
        Generate a time series that is the superposition of two periodic signals.

        Parameters:
        - T1: Period of the first sine in number of samples.
        - second_freq_multiplier: Multiplier for the period of the second sine.
        - NPeriods: Number of periods to go through, relative to the second sinusoid.
        - noise_amplitude: Amplitude of the added Gaussian noise.

        Returns:
        - time_series: An instance of TimeSeries containing the generated signal.
        """
        T2 = T1 * second_freq_multiplier  # Period of the second sine
        N = int(T2 * NPeriods)  # Total number of samples
        t = np.arange(N)  # Time indices

        # Generate the signal
        x = np.cos(2 * np.pi * (1.0 / T1) * t)  # First sinusoid
        x += np.cos(2 * np.pi * (1.0 / T2) * t)  # Second sinusoid

        # Add noise
        noise = noise_amplitude * np.random.randn(len(x))
        x += noise

        # Create and return the TimeSeries object
        return TimeSeries(t, x)

    @staticmethod
    def generate_superposed_signals(T1, T2, NPeriods, coeff1, coeff2):
        """
        Generate two time series signals (g1 and g2) that are superpositions
        of cosines with given coefficients.

        Parameters:
        - T1: Period of the first cosine in number of samples.
        - T2: Period of the second cosine in number of samples.
        - NPeriods: Number of periods to generate, based on T1.
        - coeff1: Coefficient for the first cosine.
        - coeff2: Coefficient for the second cosine.

        Returns:
        - time_series1: TimeSeries object for g1.
        - time_series2: TimeSeries object for g2.
        """
        N = T1 * NPeriods  # Total number of samples
        t = np.arange(N)   # Time indices

        # Generate g1
        x1 = coeff1 * np.cos(2 * np.pi * (1.0 / T1) * t)
        x1 += coeff2 * np.cos(2 * np.pi * (1.0 / T2) * t)
        time_series1 = TimeSeries(t, x1)

        # Generate g2
        x2 = coeff2 * np.cos(2 * np.pi * (1.0 / T1) * t)
        x2 += coeff1 * np.cos(2 * np.pi * (1.0 / T2) * t)
        time_series2 = TimeSeries(t, x2)

        return time_series1, time_series2

    def sliding_window_embedding(self, dim, tau, dt):
        """
        Return a sliding window of a time series,
        using arbitrary sampling.  Use linear interpolation
        to fill in values in windows not on the original grid

        Parameters:
        - dim: Embedding dimension.
        - tau: Delay parameter.
        - dt: Time step.

        Returns:
        - X: Embedded data matrix.
        """
        N = len(self.x)
        NWindows = int(np.floor((N - dim * tau) / dt))
        if NWindows <= 0:
            raise ValueError("Tau too large for signal extent")
        X = np.zeros((NWindows, dim))
        spl = InterpolatedUnivariateSpline(np.arange(N), self.x)
        for i in range(NWindows):
            idxx = dt * i + tau * np.arange(dim)
            if idxx[-1] >= N:
                X = X[:i, :]
                break
            X[i, :] = spl(idxx)
        return X

    def sliding_window_embedding_comp4(self, dim, tau, dt):
        N = len(self.x)
        max_time = N - (dim - 1) * tau
        times = np.arange(0, max_time, dt)
        NWindows = len(times)
        if NWindows <= 0:
            raise ValueError("Tau too large for signal extent")
        X = np.zeros((NWindows, dim))
        spl = InterpolatedUnivariateSpline(np.arange(N), self.x)
        for idx, time in enumerate(times):
            idxx = time + tau * np.arange(dim)
            if idxx[-1] >= N:
                X = X[:idx, :]
                break
            X[idx, :] = spl(idxx)
        return X

    @staticmethod
    def perform_pca_custom(data, n_components=10):
        """
        Perform PCA on the data.

        Parameters:
        - data: Data matrix to perform PCA on.
        - n_components: Number of principal components to keep.

        Returns:
        - Y: Transformed data matrix after PCA.
        - eigs: Eigenvalues of the PCA.
        """
        pca = PCA(n_components=n_components)
        Y = pca.fit_transform(data)
        eigs = pca.explained_variance_
        return Y, eigs

    @staticmethod
    def perform_pca_custom_comp(data, n_components=10):
        pca = PCA(n_components=n_components)
        Y = pca.fit_transform(data)
        eigs = pca.explained_variance_
        return {'Y': Y, 'eigenvalues': eigs}

    @staticmethod
    def generate_cosine_signal(T, NPeriods, N):
        t = np.linspace(0, 2 * np.pi * NPeriods, N+1)[:N]
        x = np.cos(t)
        return TimeSeries(t, x)

    @staticmethod
    def generate_custom_signal(t, functions):
        x = sum(f(t) for f in functions)
        return TimeSeries(t, x)

    @staticmethod
    def load_video(video_path):
        """
        Load video data from the given path and return the frames as a numpy array.
        Each row of the array corresponds to a flattened frame.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        while ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray.flatten())
            ret, frame = cap.read()
        cap.release()
        I = np.array(frames)
        frame_dims = gray.shape  # Height and width of the frames
        return I, frame_dims

    @staticmethod
    def generate_pca_video_old(I):
        """
        Perform PCA on the video data. This one consumes a lot of memory!

        Parameters:
        - I: Input data matrix where each row corresponds to a frame.

        Returns:
        - V: Transformed data after PCA.
        """
        # Subtract the mean for centering
        I_centered = I - np.mean(I, axis=0)
        # Compute covariance matrix
        ICov = np.cov(I_centered, rowvar=False)
        # Eigen decomposition
        lam, V = np.linalg.eigh(ICov)
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(lam)[::-1]
        lam = lam[idx]
        V = V[:, idx]
        # Project the data onto principal components
        I_pca = np.dot(I_centered, V)
        return I_pca

    @staticmethod
    def generate_pca_video(I, n_components=None, return_eigenvalues=False):
        """
        Perform PCA on the video data using scikit-learn's PCA with randomized SVD.

        Parameters:
        - I: Input data matrix where each row corresponds to a frame.
        - n_components: Number of principal components to keep.
        - return_eigenvalues: If True, returns the eigenvalues (explained variance).

        Returns:
        - I_pca: Transformed data after PCA.
        - eigenvalues (optional): Eigenvalues corresponding to the PCs.
        """
        # Center the data
        I_centered = I - np.mean(I, axis=0)
        # Perform PCA using randomized SVD solver
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
        I_pca = pca.fit_transform(I_centered)
        if return_eigenvalues:
            eigenvalues = pca.explained_variance_
            return I_pca, eigenvalues
        else:
            return I_pca

    @staticmethod
    def get_time_derivative(I, derivWin):
        """
        Compute the time derivative of the video data.

        Parameters:
        - I: Input data matrix where each row corresponds to a frame.
        - derivWin: Window size for computing the derivative.

        Returns:
        - X: Time derivative data.
        - validIdx: Indices of valid frames after derivative computation.
        """
        N = I.shape[0]
        validIdx = np.arange(derivWin, N - derivWin)
        X = np.zeros((len(validIdx), I.shape[1]))
        for idx, i in enumerate(validIdx):
            X[idx, :] = (I[i + derivWin, :] - I[i - derivWin, :]) / (2 * derivWin)
        return X, validIdx

    @staticmethod
    def get_sliding_window_video(I, dim, Tau, dT):
        """
        Perform sliding window embedding on video data with interpolation.

        Parameters:
        - I: Input data matrix where each row corresponds to a frame.
        - dim: Embedding dimension.
        - Tau: Time delay between embeddings.
        - dT: Time step between consecutive windows.

        Returns:
        - X: Embedded data matrix.
        """
        N = I.shape[0]  # Number of frames
        P = I.shape[1]  # Number of pixels (after PCA)
        NWindows = int(np.floor((N - dim * Tau) / dT))
        X = np.zeros((NWindows, dim * P))
        idx = np.arange(N)
        times = idx  # Original time indices

        for i in range(NWindows):
            idxx = dT * i + Tau * np.arange(dim)
            start = int(np.floor(idxx[0]))
            end = int(np.ceil(idxx[-1])) + 1
            if end >= N:
                X = X[:i, :]
                break
            # Extract the required frames for interpolation
            I_window = I[start:end+1, :]  # Shape: (number of frames in window, P)
            # Interpolate over time for each pixel
            f = interp1d(times[start:end+1], I_window, axis=0, kind='linear')
            I_interpolated = f(idxx)  # Shape: (dim, P)
            X[i, :] = I_interpolated.reshape(-1)
        return X

    @staticmethod
    def sliding_window_embedding_1d(x, dim, tau, dt):
        """
        Perform sliding window embedding on a 1D time series.

        Parameters:
        - x: 1D numpy array representing the time series.
        - dim: Embedding dimension.
        - tau: Time delay.
        - dt: Time step between windows.

        Returns:
        - X_embed: Embedded data matrix.
        """
        N = len(x)
        max_time = N - (dim - 1) * tau
        times = np.arange(0, max_time, dt)
        NWindows = len(times)
        if NWindows <= 0:
            raise ValueError("Tau too large for signal extent")
        X_embed = np.zeros((NWindows, dim))
        for idx, time in enumerate(times):
            indices = time + tau * np.arange(dim)
            indices = indices.astype(int)
            if indices[-1] >= N:
                X_embed = X_embed[:idx, :]
                break
            X_embed[idx, :] = x[indices]
        return X_embed