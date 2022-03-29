import collections

Dataset_Info = collections.namedtuple("Dataset_Info", 'data_file test_data_file categories')
class PATH(object):
    def __init__(self):
        # self.DISFA = Dataset_Info(data_file = 'create_annotation_file/DISFA/annotations.pkl',
        #  test_data_file = '',
        #     categories = {'AU': ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26'],
        #     #['AU1','AU2','AU4','AU6','AU9','AU12','AU25','AU26']
        #     })
        self.Hidden_AUs = ['AU5', 'AU9', 'AU14', 'AU16', 'AU17']

        # self.AffectNet = Dataset_Info(data_file = 'create_annotation_file/AffectNet_March/annotations_aligned_large_margin.pkl',
        #  test_data_file = '',
        #     categories = {'EXPR': ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger'],})

        # self.adjacent_EXPR = {'Happy': ['Surprise'], 
        #                       'Surprise': ['Happy', 'Fear'],
        #                       'Fear': ['Anger', 'Surprise'],
        #                       'Anger': ['Disgust', 'Fear'],
        #                       'Disgust': ['Ange,r', 'Sad'],
        #                       'Sad': ['Disgust']}
        self.ABAW3 = Dataset_Info(data_file = None, test_data_file=None,
            categories= {'AU': ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26'],
            'EXPR': ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise','Other'],
            'VA': ['valence', 'arousal']})

        
