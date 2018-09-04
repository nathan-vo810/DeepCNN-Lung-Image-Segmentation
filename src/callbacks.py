import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []


    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict([self.validation_data[0], self.validation_data[1]]))).round()
        val_targ = self.validation_data[2]
        cm1 = confusion_matrix(val_targ.argmax(axis=1), val_predict.argmax(axis=1))

        tn = np.diag(cm1)[0]
        fn = np.diag(np.fliplr(cm1))[1]
        tp = np.diag(cm1)[1]
        fp = np.diag(np.fliplr(cm1))[0]

        _val_precision = tp/(tp+fp)
        _val_recall = tp/(tp+fn)
        _val_f1 = 2*(_val_recall*_val_precision)/(_val_recall + _val_precision)

        #
        # _val_f1 = f1_score(val_targ, val_predict, average='micro')
        # _val_recall = recall_score(val_targ, val_predict, average='micro')
        # _val_precision = precision_score(val_targ, val_predict, average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return