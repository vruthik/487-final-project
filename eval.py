from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

class Evaluation():
    """Evaluation metrics."""
    def __init__(self, pred, true):
        self.pred = pred
        self.true = true
    
    def acc(self):
        score = accuracy_score(self.true, self.pred)
        print(f'Accuracy: {score}')

    def f1_score_micro(self):
        score = f1_score(self.true, self.pred, average='micro')
        print(f'F1 score micro: {score}')
    def f1_score_macro(self):
        score = f1_score(self.true, self.pred, average='macro')
        print(f'F1 score macro: {score}')

    def precision_micro(self):
        score = precision_score(self.true, self.pred, average='micro')
        print(f'Precision micro: {score}')
    def precision_macro(self):
        score = precision_score(self.true, self.pred, average='macro')
        print(f'Precision macro: {score}')

    def recall_micro(self):
        score = recall_score(self.true, self.pred, average='micro')
        print(f'Recall micro: {score}')
    def recall_macro(self):
        score = recall_score(self.true, self.pred, average='macro')
        print(f'Recall macro: {score}')    

    def all_metrics(self):
        self.acc()
        print("-----------------------")
        print("Macro")
        self.f1_score_macro()
        self.precision_macro()
        self.recall_macro()
        print("-----------------------")
        print("Micro")
        self.f1_score_micro()
        self.precision_micro()
        self.recall_micro()

