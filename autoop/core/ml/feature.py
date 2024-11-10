class Feature:
    """
    A class that represents a feature.
    """
    def __init__(self, name: str, type: str) -> None:
        """
        Initialezes a Feature object with a name and type.

        Args:
            name (str): The name of the feature.
            type (str): The type of the feature.
        """
        self._name = name
        self._type = type
    
    @property
    def name(self) -> str:
        """
        Public getter for the name attribute.
        """
        return self._name   
     
    @property
    def type(self) -> str:
        """
        Public getter for the type attribute.
        """
        return self._type   

    def __str__(self) -> str:
        """
        String with the Feature object.
        Returns:
            str: A string with the name and type of the Feature.
        """
        return f"Feature(name={self._name}, type={self._type})"
