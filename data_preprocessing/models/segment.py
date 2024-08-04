
class Segment:
    ''' Represents a segment of an input audio when 1 speaker speaks: 
        speaker (label); start:end 
    '''
        
    def __init__(self, start, end, speaker):
        self.start = start #ms
        self.end = end #ms
        self.speaker = speaker

    def __str__(self):
        return f"Segment(\n{self.start} - {self.end}, {self.speaker},\n)"