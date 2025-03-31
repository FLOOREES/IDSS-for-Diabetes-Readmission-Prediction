"""
Steps:
- Convert Preprocessing into a class with methods for each step. So we can perform different steps independently, in order to obtain different preprocessed datasets.
- Convert the Oultiers notebooks into another class.
- Combine the two classes into a single pipeline class (this one).
- Program a adavanced outlier detection method (a type of AutoEncoder optionally combined with an Isolation Forest). So we can apply a stathistics based method 
to a not i.i.d dataframe.
"""

import pandas as pd

# TODO

class Outliers:
    """
    Class for outlier detection and removal from a DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def detect_outliers(self, method: str = 'z_score', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in the DataFrame using the specified method.

        Parameters
        ----------
        method : str
            Method to use for outlier detection. Options are 'z_score' or 'iqr'.
        threshold : float
            Threshold for outlier detection.

        Returns
        -------
        pd.DataFrame
            DataFrame with outliers removed.
        """
        if method == 'z_score':
            return self._remove_outliers_z_score(threshold)
        elif method == 'iqr':
            return self._remove_outliers_iqr(threshold)
        else:
            raise ValueError("Invalid method. Use 'z_score' or 'iqr'.")