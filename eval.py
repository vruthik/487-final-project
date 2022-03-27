from sklearn.metrics import f1_score, accuracy_score

class Evaluation():
    """Evaluation metrics."""
    def __init__(self, pred, true):
        self.pred = pred
        self.true = true
    def f1_score_micro(self):
        score = f1_score(self.true, self.pred, average='micro')
        print(f'f1 score micro: {score}')
    def f1_score_macro(self):
        score = f1_score(self.true, self.pred, average='macro')
        print(f'f1 score macro: {score}')
    def acc(self):
        score = accuracy_score(self.true, self.pred)
        print(f'acc: {score}')


