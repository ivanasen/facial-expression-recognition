
def get_expression_label(result):
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    m = 0.000000000000000000001
    a = result[0]
    for i in range(0, len(a)):
        if a[i] > m:
            m = a[i]
            ind = i

    return emotions[ind]
