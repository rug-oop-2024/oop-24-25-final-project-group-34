from typing import List, Union
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd
import io


def detect_feature_types(dataset: Union[pd.DataFrame,
                                        Dataset]) -> List[Feature]:
    """Assumption: only categorical and numerical features
    and no NaN values.
    Args:
        dataset(Union[pd.DataFrame, Dataset]): Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    data = dataset.read()

    if isinstance(data, pd.DataFrame):
        data = data
    else:
        data = pd.read_csv(io.StringIO(data.decode("utf-8")))

    for column in data.columns:
        if all(isinstance(value, (float, int))
               for value in data[column]):
            type_feature = "numerical"
        else:
            type_feature = "categorical"

        features.append(Feature(name=column,
                                type=type_feature))

    return features
