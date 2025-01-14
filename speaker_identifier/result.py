class Result(dict[str, float]):
    """
    Represents the prediction result for a single audio sample.

    This class is used to store and manage the probabilities or scores associated 
    with different labels or speakers for a single audio sample. Each key-value 
    pair corresponds to a label or speaker and its corresponding prediction probability.

    Inherits:
        dict[str, float]: A dictionary where keys are strings (labels or speakers) 
                          and values are floating-point probabilities or scores.

    Methods:
        __init__(*args, **kwargs): Initializes the Result object by calling the parent 
                                  class constructor with any given arguments or keyword arguments.
    """
    def __init__(self, *args, **kwargs):
        super(Result, self).__init__(*args, **kwargs)
        
        
        
    @property
    def best_prediction(self) -> str:
        return max(self, key=self.get)
        