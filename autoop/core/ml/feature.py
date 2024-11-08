from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature:
    def __init__(self, name: str, type: str) -> None:
        self._name = name
        self._type = type
    
    @property
    def name(self) -> str:
        """Public getter for the name attribute."""
        return self._name   
     
    @property
    def type(self) -> str:
        """Public getter for the type attribute."""
        return self._type   

    def __str__(self) -> str:
        return f"Feature(name={self._name}, type={self._type})"
