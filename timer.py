        
import time

#Class for measuring times of execution!
class Timer:

    def start_training(self):
        self.start_training = time.time()

    def end_training(self):
        self.training_time = time.time()-self.start_training
        
    def start_prepare_train(self):
        self.start_prepare_train = time.time()

    def end_prepare_train(self):
        self.prepare_train_time = time.time()-self.start_prepare_train
        
    def start_prepare_test(self):
        self.start_prepare_test = time.time()

    def end_prepare_test(self):
        self.prepare_test_time = time.time()-self.start_prepare_test

    def start_predict(self):
        self.start_predict = time.time()

    def end_predict(self):
        self.predict_time = time.time()-self.start_predict
        
    def start_execution(self):
        self.start_predict = time.time()

    def end_execution(self):
        self.execution_time = time.time()-self.start_execution
        
    def __repr__(self):
        return f"""
        Execution {self.execution_time} \n
        Train: {self.training_time} \n
        Prepare train: {self.prepare_train_time} \n
        Prepare test: {self.prepare_test_time} \n
        Predict: {self.predict_time} \n
        """
        
    