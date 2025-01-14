class Result(dict[str, dict[str, float]]):
    def __init__(self, *args, **kwargs):
        super(Result, self).__init__(*args, **kwargs)