import numpy as np
import gazenetGenerator2 as gaze_gen


def compare(model, test_data, num_dataset):
    err_list = []   # idx, truth, predict

    for i in range(num_dataset):
        input, truth = next(test_data)
        # print(img_seq.shape)
        # print(gaze_seq.shape)
        # print(truth.shape)
        predict = model.predict(input, batch_size=1)
        truth_idx = np.argmax(truth[0])
        predict_idx = np.argmax(predict[0])
        # print(truth[0])
        # print(predict[0])
        # print(truth_idx)
        # print(predict_idx)
        if truth_idx != predict_idx:
            err_list.append([i, predict[0], truth[0]])
    print("accuracy: " + str(1 - len(err_list)/float(num_dataset)))
    return err_list

def compare1(model, test_data, num_dataset):
    err_list = []   # idx, truth, predict

    for i in range(num_dataset):
        input, truth = next(test_data)
        [img_seq, gaze_seq] = input
        # predict = model.predict([img_seq, gaze_seq], batch_size=1)
        predict = model.predict([img_seq, gaze_seq], batch_size=None, steps=1)
        truth_idx = np.argmax(truth[0])
        predict_idx = np.argmax(predict[0])
        # print(truth[0])
        # print(predict[0])
        # print(truth_idx)
        # print(predict_idx)
        if truth_idx != predict_idx:
            err_list.append([i, predict[0], truth[0]])
    print("accuracy: " + str(1 - len(err_list)/float(num_dataset)))
    return err_list
