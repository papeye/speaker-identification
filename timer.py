        
import time

#Class for measuring times of execution!
class Timer:

    def __init__(self):
        self.training_time = None
        self.prepare_train_time = None
        self.prepare_test_time = None
    
    def start_training(self):
        self.start_train = time.time()

    def end_training(self):
        self.training_time = time.time()-self.start_train
        
    def start_prepare_train(self):
        self.start_prepare = time.time()

    def end_prepare_train(self):
        self.prepare_train_time = time.time()-self.start_prepare
        
    def start_prepare_test(self):
        self.start_prepare= time.time()

    def end_prepare_test(self):
        self.prepare_test_time = time.time()-self.start_prepare

    def start_predicting(self):
        self.start_predict = time.time()

    def end_predict(self):
        self.predict_time = time.time()-self.start_predict
        
    def start_executing(self):
        self.start_execution = time.time()

    def end_execution(self):
        self.execution_time = time.time()-self.start_execution
        
    def __repr__(self):
        return f"""
        Execution {self.execution_time} \n
        Train: {self.training_time if self.training_time is not None else 0} \n
        Prepare train: {self.prepare_train_time if self.prepare_train_time is not None else 0} \n
        Prepare test: {self.prepare_test_time if self.prepare_test_time is not None else 0} \n
        Predict: {self.predict_time} \n
        """
        
    