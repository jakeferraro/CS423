from __future__ import annotations  #must be first line in your library!
import pandas as pd
import numpy as np
import types
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sklearn
from sklearn import set_config
set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices



class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
       category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  #column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
        warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

        #now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        #now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result



class CustomOHETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column: str, dummy_na: bool = False, drop_first: bool = False) -> None:
        self.target_column = target_column
        self.dummy_na = dummy_na
        self.drop_first = drop_first

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'

        # Handle NaN values before applying get_dummies
        X[self.target_column] = X[self.target_column].astype(str)

        # Create dummy variables and add them to the DataFrame
        X_ = pd.concat([X, pd.get_dummies(X[self.target_column],
                                          prefix=self.target_column,
                                          prefix_sep='_',
                                          dummy_na=self.dummy_na,
                                          drop_first=self.drop_first,
                                          dtype=int)], axis=1)

        # Remove the original target column
        X_ = X_.drop(columns=[self.target_column])

        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result



class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It allows for selectively keeping or dropping columns
    from a DataFrame based on a provided list.

    Parameters
    ----------
    column_list : List[str]
        List of column names to either drop or keep, depending on the action parameter.
    action : str, default='drop'
        The action to perform on the specified columns. Must be one of:
        - 'drop': Remove the specified columns from the DataFrame
        - 'keep': Keep only the specified columns in the DataFrame

    Attributes
    ----------
    column_list : List[str]
        The list of column names to operate on.
    action : str
        The action to perform ('drop' or 'keep').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>>
    >>> # Drop columns example
    >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
    >>> dropped_df = dropper.fit_transform(df)
    >>> dropped_df.columns.tolist()
    ['C']
    >>>
    >>> # Keep columns example
    >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
    >>> kept_df = keeper.fit_transform(df)
    >>> kept_df.columns.tolist()
    ['A', 'C']
    """

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        """
        Initialize the CustomDropColumnsTransformer.

        Parameters
        ----------
        column_list : List[str]
            List of column names to either drop or keep.
        action : str, default='drop'
            The action to perform on the specified columns.
            Must be either 'drop' or 'keep'.

        Raises
        ------
        AssertionError
            If action is not 'drop' or 'keep', or if column_list is not a list.
        """
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list: List[str] = column_list
        self.action: Literal['drop', 'keep'] = action

    #your code below
    def fit(self, X: pd.DataFrame, y: None = None) -> "CustomDropColumnsTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the column dropping or keeping operation to the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with the specified columns dropped or kept.
        """
        if self.action == 'drop':
            return X.drop(columns=self.column_list, errors='ignore')
        elif self.action == 'keep':
            return X[[col for col in X.columns if col in self.column_list]]
        else:
            raise ValueError(f"Invalid action '{self.action}'. Must be 'drop' or 'keep'.")



titanic_transformer = Pipeline(steps=[
    ('gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('ohe_joined', CustomOHETransformer(target_column='Joined')),
    ('fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ], verbose=True)



customer_transformer = Pipeline(steps=[
    ('drop_columns', CustomDropColumnsTransformer(column_list=['ID'], action='drop')),
    ('gender', CustomMappingTransformer('Gender', {'Female': 1, 'Male': 0})),
    ('experience_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high': 2})),
    ('os', CustomOHETransformer(target_column='OS')),
    ('isp', CustomOHETransformer(target_column='ISP')),
    ], verbose=True)



class CustomPearsonTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that removes highly correlated features
    based on Pearson correlation.

    Parameters
    ----------
    threshold : float
        The correlation threshold above which features are considered too highly correlated
        and will be removed.

    Attributes
    ----------
    correlated_columns : Optional[List[Hashable]]
        A list of column names (which can be strings, integers, or other hashable types)
        that are identified as highly correlated and will be removed.
    """
    def __init__(self, threshold=0.4):
        self.threshold = threshold
        self.correlated_columns = None

    def fit(self, X, y=None):
        df_corr = X.corr(method='pearson')
        abs_corr = np.abs(df_corr)
        masked_df = abs_corr > self.threshold
        upper_mask = np.triu(np.ones(masked_df.shape), k=1).astype(bool)
        masked_df = masked_df & upper_mask
        self.correlated_columns = [masked_df.columns[i] for i, col in enumerate(masked_df.astype(bool).T) if masked_df.astype(bool).iloc[:, i].any()]
        return self

    def transform(self, X):
        if self.correlated_columns is None:
            raise AssertionError("PearsonTransformer.transform called before fit.")
        return X.drop(columns=self.correlated_columns)



class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """
    def __init__(self, target_column: Hashable):
        self.target_column = target_column
        self.high_wall = None
        self.low_wall = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the transformer to the data by computing the 3-sigma boundaries.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns self.
        """

        assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        assert pd.api.types.is_numeric_dtype(X[self.target_column]), f'expected int or float in column {self.target_column}'

        # Compute the mean and standard deviation of the target column
        mean = X[self.target_column].mean()
        sigma = X[self.target_column].std()

        # Compute the low and high boundaries
        self.low_wall = mean - 3 * sigma
        self.high_wall = mean + 3 * sigma

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by clipping the target column.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with clipped values.
        """

        assert self.high_wall is not None and self.low_wall is not None, 'Sigma3Transformer.fit has not been called.'

        # Clip the target column
        X_clipped = X.copy()  # Create a copy to avoid modifying the original DataFrame
        X_clipped[self.target_column] = X_clipped[self.target_column].clip(lower=self.low_wall, upper=self.high_wall)
        
        # Reset index
        X_clipped = X_clipped.reset_index(drop=True)
        
        return X_clipped



class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
    It clips values in the target column based on Tukey's inner or outer fences.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply Tukey's fences on.
    fence : Literal['inner', 'outer'], default='outer'
        Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).

    Attributes
    ----------
    inner_low : Optional[float]
        The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
    outer_low : Optional[float]
        The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
    inner_high : Optional[float]
        The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
    outer_high : Optional[float]
        The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
    >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
    >>> transformed_df = tukey_transformer.fit_transform(df)
    >>> transformed_df
    """
    def __init__(self, target_column: Hashable, fence: Literal['inner', 'outer'] = 'outer'):
        self.target_column = target_column
        self.fence = fence
        self.inner_low = None
        self.outer_low = None
        self.inner_high = None
        self.outer_high = None
        

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the transformer to the data by computing Tukey's fences.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns self.
        """

        assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'unknown column {self.target_column}'
        assert pd.api.types.is_numeric_dtype(X[self.target_column]), f'expected int or float in column {self.target_column}'

        # Compute quartiles and IQR
        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)
        iqr = q3 - q1

        # Compute inner and outer fences
        self.inner_low = q1 - 1.5 * iqr
        self.inner_high = q3 + 1.5 * iqr
        self.outer_low = q1 - 3 * iqr
        self.outer_high = q3 + 3 * iqr

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by clipping the target column based on Tukey's fences.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with clipped values.
        """
        
        assert self.inner_low is not None and self.inner_high is not None, 'TukeyTransformer.fit has not been called.'
        assert self.outer_low is not None and self.outer_high is not None, 'TukeyTransformer.fit has not been called.'

        # Clip the target column based on the specified fence
        X_clipped = X.copy()  # Create a copy to avoid modifying the original DataFrame
        
        if self.fence == 'inner':
            X_clipped[self.target_column] = X_clipped[self.target_column].clip(lower=self.inner_low, upper=self.inner_high)
        elif self.fence == 'outer':
            X_clipped[self.target_column] = X_clipped[self.target_column].clip(lower=self.outer_low, upper=self.outer_high)

        # Reset index
        X_clipped = X_clipped.reset_index(drop=True)
        
        return X_clipped

