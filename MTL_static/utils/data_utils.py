import torchvision.transforms as transforms
import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PATH import PATH
PRESET_VARS  = PATH()
EXPR_categories = PRESET_VARS.ABAW3.categories['EXPR']
def averaged_f1_score(input, target):
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        f1 = f1_score(input[:, i], target[:, i])
        f1s.append(f1)
    return np.mean(f1s), f1s
def accuracy(input, target):
    assert len(input.shape) == 1
    return sum(input==target)/input.shape[0]
def averaged_accuracy(x, y):
    assert len(x.shape) == 2
    N, C =x.shape
    accs = []
    for i in range(C):
        acc = accuracy(x[:, i], y[:, i])
        accs.append(acc)
    return np.mean(accs), accs
def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def VA_metric(x, y):
    items = [CCC_score(x[:,0], y[:,0]), CCC_score(x[:,1], y[:,1])]
    return items, sum(items)
def EXPR_metric(x, y): 
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average= 'macro')
    acc = accuracy(x, y)
    return [f1, acc], 0.67*f1 + 0.33*acc
def AU_metric(x, y):
    f1_av, f1s  = averaged_f1_score(x, y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    acc_av  = accuracy(x, y)
    return [f1_av, acc_av, f1s], 0.5*f1_av + 0.5*acc_av
EPS = 1e-8
class CCCLoss(nn.Module):
    def __init__(self, digitize_num=20, range=[-1, 1], weight=None):
        super(CCCLoss, self).__init__() 
        self.digitize_num =  digitize_num
        self.range = range
        self.weight = weight
        if self.digitize_num >1:
            bins = np.linspace(*self.range, num= self.digitize_num)
            self.bins = torch.as_tensor(bins, dtype = torch.float32).cuda().view((1, -1))
    def forward(self, x, y): 
        # the target y is continuous value (BS, )
        # the input x is either continuous value (BS, ) or probability output(digitized)
        y = y.view(-1)
        if self.digitize_num !=1:
            x = F.softmax(x, dim=-1)
            x = (self.bins * x).sum(-1) # expectation
        x = x.view(-1)
        if self.weight is None:
            vx = x - torch.mean(x) 
            vy = y - torch.mean(y) 
            rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + EPS)
            x_m = torch.mean(x)
            y_m = torch.mean(y)
            x_s = torch.std(x)
            y_s = torch.std(y)
            ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + EPS)
        else:
            rho = weighted_correlation(x, y, self.weight)
            x_var = weighted_covariance(x, x, self.weight)
            y_var = weighted_covariance(y, y, self.weight)
            x_mean = weighted_mean(x, self.weight)
            y_mean = weighted_mean(y, self.weight)
            ccc = 2*rho*torch.sqrt(x_var)*torch.sqrt(y_var)/(x_var + y_var + torch.pow(x_mean - y_mean, 2) +EPS)
        return 1-ccc
        
def get_metric_func(task):
    if task =='VA':
        return VA_metric
    elif task=='EXPR':
        return EXPR_metric
    elif task=='AU':
        return AU_metric
        
def train_transforms(img_size):
    transform_list = [transforms.Resize([int(img_size*1.02), int(img_size*1.02)]),
                      transforms.RandomCrop([img_size, img_size]),
                      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
                    ]
    return transforms.Compose(transform_list)

def test_transforms(img_size):
    transform_list = [transforms.Resize([img_size, img_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
                    ]
    return transforms.Compose(transform_list)

class Resize_with_Attention(transforms.Resize):
    def __init__(self, *args, **kwargs):
        super(Resize_with_Attention, self).__init__(*args, **kwargs)
        
    def forward(self, input):
        img, attention = input
        return (TF.resize(img, self.size), TF.resize(attention, self.size))
class RandomCrop_with_Attention(transforms.RandomCrop):
    def __init__(self, *args, **kwargs):
        super(RandomCrop_with_Attention, self).__init__(*args, **kwargs)
    def forward(self, input):
        img, attention = input
        if self.padding is not None:
            img = TF.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = TF._get_image_size(img)
        w1, h1 = TF._get_image_size(attention)
        assert width==w1 and height==h1, "attention map has to be the same size with the input image"
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return (TF.crop(img, i, j, h, w), TF.crop(attention, i, j, h, w))

class RandomHorizontalFlip_with_Attention(transforms.RandomHorizontalFlip):
    def __init__(self, *args, **kwargs):
        super(RandomHorizontalFlip_with_Attention, self).__init__(*args, **kwargs)
    def forward(self, input):
        img, attention = input
        if torch.rand(1) < self.p:
            return (TF.hflip(img), TF.hflip(attention))
        else:
            return (img, attention)

def visualize_attention_maps(image, attention):
    import matplotlib.pyplot as plt
    N_aus = attention.size(0)
    fig, axes = plt.subplots(1, N_aus+1, figsize=((N_aus+1)*5, 5))
    axes[0].imshow(image)
    axes[0].axis('off')
    x, y = np.arange(-5.0, 5.0, 0.05), np.arange(-5.0, 5.0,  0.05)
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    for i_au in range(N_aus):
        atten = attention[i_au]
        if atten.is_cuda:
            atten = atten.cpu()
        atten = atten.numpy()
        im1 = axes[i_au+1].imshow(image, cmap=plt.cm.gray, alpha=0.5, interpolation='nearest',
            extent =extent)
        im2 = axes[i_au+1].imshow(atten, cmap = plt.cm.viridis, alpha=0.5, interpolation='bilinear',
            extent = extent)
        axes[i_au+1].axis('off')
    plt.show()
def train_transforms_with_attention(img_size, attention_size=17):
    transform_listA = [Resize_with_Attention([int(img_size*1.02), int(img_size*1.02)]),
                      RandomCrop_with_Attention([img_size, img_size]),
                      RandomHorizontalFlip_with_Attention()
                    ]
    transformA = transforms.Compose(transform_listA)

    transform_listB = [
                      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
                      ]
    transformB = transforms.Compose(transform_listB)
    def func(image, attention):
        image, attention = transformA((image, attention))
        attention = transforms.Resize(attention_size)(attention)
        # import pdb; pdb.set_trace()
        # start visualization
        # visualize_attention_maps(image, attention)
        # end visualization
        image = transformB(image)
        return image, attention
    return func
def test_transforms_with_attention(img_size, attention_size=17):
    transform_listA = [Resize_with_Attention([img_size, img_size])
                    ]
    transformA = transforms.Compose(transform_listA)

    transform_listB = [
                      transforms.ToTensor(),
                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
                      ]
    transformB = transforms.Compose(transform_listB)
    def func(image, attention):
        image, attention = transformA((image, attention))
        attention = transforms.Resize(attention_size)(attention)
        image = transformB(image)
        return image, attention
    return func

def inverse_test_transform(x):
    std = torch.ones(3).to(x.device).unsqueeze(-1).unsqueeze(-1)
    std.index_fill_(0, torch.tensor([0]), 0.229)
    std.index_fill_(0, torch.tensor([1]), 0.224)
    std.index_fill_(0, torch.tensor([2]), 0.225)
    mean = torch.zeros(3).to(x.device).unsqueeze(-1).unsqueeze(-1)
    mean.index_fill_(0, torch.tensor([0]), 0.485)
    mean.index_fill_(0, torch.tensor([1]), 0.456)
    mean.index_fill_(0, torch.tensor([2]), 0.406)

    x = x *std + mean
    x = x*225
    return x
def extact_face_landmarks(df):
    face_shape = []
    for i in range(1, 69):
        x,  y = df['x_{}'.format(i)], df['y_{}'.format(i)]
        x, y = int(x), int(y)
        face_shape.append([x, y])
    return np.array(face_shape)

def transform_corrdinate(x, y, in_size, out_size):
    new_x, new_y = x * (out_size[0]/in_size[0]), y * (out_size[1]/in_size[1])
    new_x, new_y = int(new_x), int(new_y)
    assert new_x< out_size[0] and new_y< out_size[1]
    return new_x, new_y

def landmarks_to_attention_map(AU_name, face_ldm, in_size = (224, 224), out_size = (112, 112)):
    AU_dict = {'AU1': [22, 23], 'AU2': [19, 26],
    'AU4': [22, 23, 39, 44], 'AU5': [38, 39, 44, 45, 41, 42, 47, 48],
    'AU6': [47, 48, 41, 42], 'AU9': [28, 29, 30, 32, 36, 22, 23],
    'AU10': [51, 53, 32, 36], 'AU12': [49, 55, 52, 58],
    'AU14': [49, 55, 52, 58], 'AU15': [49, 55, 52, 58],
    'AU17':[49, 55, 57, 59], 'AU20':[49, 55, 57, 59], 
    'AU25': [50, 52, 54, 56, 58, 60],
    'AU26': [50, 52, 54, 56, 58, 60, 8, 9, 10]}
    landmarks_list = AU_dict[AU_name]
    assert in_size[0] == in_size[1] and in_size[0]==224, "input image size and face landmarks are for [224, 224] images"
    out = np.zeros(out_size)
    for ldm_id in landmarks_list:
        x, y = face_ldm[ldm_id-1]
        new_x, new_y = transform_corrdinate(x,y, in_size, out_size)
        out_ldm = fit_ellipse_to_landmark(new_x, new_y, out_size)
        out += out_ldm
    offset = face_ldm[42-1][1] - face_ldm[20-1][1]  # the y diff between eye brow and lower eye lid
    assert offset >0, "offset positive"
    if AU_name == 'AU1':
        for ldm_id in [22, 23]:
            x, y = face_ldm[ldm_id-1]
            y = max(0, y-offset)
            new_x, new_y = transform_corrdinate(x,y, in_size, out_size)
            out += fit_ellipse_to_landmark(new_x, new_y, out_size)
    elif AU_name =='AU2':
        for ldm_id in [19, 26]:
            x, y = face_ldm[ldm_id-1]
            y = max(0, y-offset)
            new_x, new_y = transform_corrdinate(x,y, in_size, out_size)
            out += fit_ellipse_to_landmark(new_x, new_y, out_size)
    elif AU_name =='AU4':
        x1, y1 = face_ldm[20-1]
        x2, y2 = face_ldm[38-1]
        x, y = (x1+x2)*0.5, (y1+y2)*0.5
        new_x, new_y = transform_corrdinate(x,y, in_size, out_size)
        out += fit_ellipse_to_landmark(new_x, new_y, out_size)
        x1, y1 = face_ldm[25-1]
        x2, y2 = face_ldm[45-1]
        x, y = (x1+x2)*0.5, (y1+y2)*0.5
        new_x, new_y = transform_corrdinate(x,y, in_size, out_size)
        out += fit_ellipse_to_landmark(new_x, new_y, out_size)
    elif AU_name =='AU6':
        x, y = face_ldm[42-1][0], face_ldm[32-1][1]
        new_x, new_y = transform_corrdinate(x,y, in_size, out_size)
        out += fit_ellipse_to_landmark(new_x, new_y, out_size)
        x, y = face_ldm[47-1][0], face_ldm[36-1][1]
        new_x, new_y = transform_corrdinate(x,y, in_size, out_size)
        out += fit_ellipse_to_landmark(new_x, new_y, out_size)
        x, y = face_ldm[42-1][0], face_ldm[31-1][1]
        new_x, new_y = transform_corrdinate(x,y, in_size, out_size)
        out += fit_ellipse_to_landmark(new_x, new_y, out_size)
        x, y = face_ldm[47-1][0], face_ldm[31-1][1]
        new_x, new_y = transform_corrdinate(x,y, in_size, out_size)
        out += fit_ellipse_to_landmark(new_x, new_y, out_size)
    elif AU_name =='AU17' or AU_name =='AU26' or AU_name =='AU20':
        x1, y1 = face_ldm[8-1]
        x2, y2 = face_ldm[59-1]
        x, y = (x1+x2)*0.5, (y1+y2)*0.5
        new_x, new_y = transform_corrdinate(x,y, in_size, out_size)
        out += fit_ellipse_to_landmark(new_x, new_y, out_size)
        x1, y1 = face_ldm[10-1]
        x2, y2 = face_ldm[57-1]
        x, y = (x1+x2)*0.5, (y1+y2)*0.5
        new_x, new_y = transform_corrdinate(x,y, in_size, out_size)
        out += fit_ellipse_to_landmark(new_x, new_y, out_size)

    out = blurry_image(out)
    out = out/out.max() # 0 1to 1
    return out

def blurry_image(input_img):
    return gaussian_filter(input_img, sigma=3)
def fit_ellipse_to_landmark(x, y, out_size = (224, 224), sigma=20):
    N = out_size[0]
    assert N == out_size[1], "Square image only"
    X = np.arange(N) - x
    Y = np.arange(N) - y
    X, Y = np.meshgrid(X, Y)
    cov = np.array([[sigma*1., 0.], [0., sigma*1.]])
    pos = np.empty(X.shape+ (2,))
    pos[:, :, 0] = X 
    pos[:, :, 1] = Y

    out = multivariate_gaussian(pos, np.array([0, 0]), cov)
    return out
def multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n*Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac/2)/N
class FocalLoss(nn.Module):
    """
    Loss(x, class) = - \alpha(1-p(x)[class])^gamma \log(p(x)[class])
    """
    def __init__(self, class_num, alpha = None, gamma = 2, size_average = True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha 
            else:
                self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.new(N, C).fill_(0)
        ids = targets.view(-1, 1).long()
        class_mask.scatter_(1, ids, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.view(-1)]
        probas = (P*class_mask).sum(1).view(-1)
        log_p = probas.log()
        batch_loss = -alpha *(torch.pow((1-probas), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class Tversky_Loss_with_Logits(nn.Module):
    def __init__(self, beta=0.7, pos_weight = None, reduction='mean'):
        super(Tversky_Loss_with_Logits, self).__init__()
        self.beta = beta
        self.pos_weight = pos_weight
        if self.pos_weight is not None:
            if not isinstance(self.pos_weight, torch.Tensor):
                raise ValueError("pos weight should be an tensor!")
        self.reduction = reduction
    def forward(self, inputs, targets):
        #import pdb; pdb.set_trace()
        batch_size = targets.size(0)
        inputs = torch.sigmoid(inputs)
        tp = inputs*targets
        fn = (1 - inputs)* targets
        fp = inputs* (1-targets)
        alpha = 1- self.beta
        x = alpha *fp + self.beta*fn
        eps = 1e-9
        if self.pos_weight is not None:
            if inputs.is_cuda and not self.pos_weight.is_cuda:
                self.pos_weight = self.pos_weight.cuda()
            tversky_loss = (self.pos_weight.unsqueeze(0)*x + eps)/(tp +x + eps)
        else:
            tversky_loss = (x+ eps)/(tp+x+ eps)
        if self.reduction =='mean':
            return tversky_loss.mean()
        elif self.reduction=='sum':
            return tversky_loss.sum()
        else:
            return tversky_loss
class FocalTversky_Loss_with_Logits(nn.Module):
    def __init__(self, beta=0.7, gamma=1., pos_weight = None, reduction='mean'):
        super(FocalTversky_Loss_with_Logits, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.pos_weight = pos_weight
        if self.pos_weight is not None:
            if not isinstance(self.pos_weight, torch.Tensor):
                raise ValueError("pos weight should be an tensor!")
        self.reduction = reduction
    def forward(self, inputs, targets):
        #import pdb; pdb.set_trace()
        batch_size = targets.size(0)
        inputs = torch.sigmoid(inputs)
        tp = inputs*targets
        fn = (1 - inputs)* targets
        fp = inputs* (1-targets)
        alpha = 1- self.beta
        x = alpha *fp + self.beta*fn
        eps = 1e-9
        if self.pos_weight is not None:
            if inputs.is_cuda and not self.pos_weight.is_cuda:
                self.pos_weight = self.pos_weight.cuda()
            tversky_loss = (self.pos_weight.unsqueeze(0)*x + eps)/(tp +x + eps)
        else:
            tversky_loss = (x+ eps)/(tp+x+ eps)
        tversky_loss = torch.pow(tversky_loss, self.gamma)
        if self.reduction =='mean':
            return tversky_loss.mean()
        elif self.reduction=='sum':
            return tversky_loss.sum()
        else:
            return tversky_loss

def compute_center_contrastive_loss(features, centers, targets):
    """Compute Center contrastive loss
    
    Args:
        features (torch.Tensor): size (Bs, Num_classes, Emb_size)
        centers (torch.Tensor): size (Num_classes, Emb_size)
        targets (torch.Tensor): size (Bs, Num_classes) binary labels
    
    Returns:
        TYPE: Description
    """
    criterion = torch.nn.MSELoss()
    Num_classes = features.size(1)
    pos = 0
    neg = 1e-9 # epsilon
    for i_c in range(Num_classes):
        pos_mask = targets[..., i_c]==1
        neg_mask = targets[..., i_c]==0
        class_center = centers[i_c]
        pos_feature = features[pos_mask, i_c]
        N_pos = sum(pos_mask)
        if N_pos>0:
            pos += criterion(pos_feature, class_center.unsqueeze(0).repeat(N_pos, 1))
        neg_feature = features[neg_mask, i_c]
        N_neg = sum(neg_mask)
        if N_neg >0:
            neg += criterion(neg_feature, class_center.unsqueeze(0).repeat(N_neg, 1))
    if isinstance(pos, torch.Tensor):
        pos = pos.sum()
    if isinstance(neg, torch.Tensor):
        neg = neg.sum()
    return pos/neg

def get_center_delta(features, centers, targets, alpha):
    """Compute delta center
    
    Args:
        features (torch.Tensor): size (Bs, Num_classes, Emb_size)
        centers (torch.Tensor): size (Num_classes, Emb_size)
        targets (torch.Tensor): size (Bs, Num_classes) binary labels
        alpha (float): the coefficient to be multiplied with the delta center
    
    Returns:
        TYPE: Description
    """
    # implementation equation (4) in the center-loss paper
    Bs = features.size(0)
    Num_classes = features.size(1)
    Emb_size = features.size(2)
    delta_centers = torch.zeros(Num_classes, Emb_size).to(features.device)

    for i_c in range(Num_classes):
        pos_mask = targets[:, i_c]==1
        pos_sum = sum(pos_mask)
        if pos_sum>0:
            delta = (centers[i_c].unsqueeze(0).repeat(Bs, 1) - features[:, i_c])[pos_mask]
            delta_centers[i_c, :] = delta.sum(0)/(1.+pos_sum) * alpha
    return delta_centers



# get the triplet index from regression labels
def get_ap_and_an(va_labels, delta_ap, delta_an, ref_labels=None):
    if ref_labels is None:
        ref_labels = va_labels
    x1, y1 = va_labels[:, 0], va_labels[:, 1]
    x2, y2 = ref_labels[:, 0], ref_labels[:, 1]
    euclidean_distance = torch.sqrt((x2.unsqueeze(1) - x1.unsqueeze(0))**2 + (y2.unsqueeze(1) - y1.unsqueeze(0))**2)

    matches = (euclidean_distance <=delta_ap).byte()
    diffs = (euclidean_distance >=delta_an).byte()
    if ref_labels is va_labels:
        matches.fill_diagonal_(0)

    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)


def get_matches_and_diffs(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs
def get_adjacent_and_non_adjacent(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    N = len(labels)
    adjacents = torch.zeros((N, N)).to(labels.device)
    for i_row in range(N):
        for i_col in range(N):
            x, y = labels[i_row], labels[i_col]
            adjacents[i_row, i_col] = is_adajcent(x,y)
    adjacents = adjacents.byte()
    nonadjacents = adjacents^ 1
    if ref_labels is labels:
        nonadjacents.fill_diagonal_(0)
    return adjacents, nonadjacents

def is_adajcent(x, y):
    adjacent_EXPR = PRESET_VARS.adjacent_EXPR
    name = EXPR_categories[x]
    adjacent_names = adjacent_EXPR[name]
    for a_name in adjacent_names:
        a_index = EXPR_categories.index(a_name)
        if y==a_index:
            return True
    return False

def get_all_triplets_indices(labels, ref_labels=None):
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    triplets = matches.unsqueeze(2) * diffs.unsqueeze(1)
    return torch.where(triplets)

def get_quadruplet(expr_labels, ref_labels = None):
    assert EXPR_categories.index('Neutral') not in expr_labels, "sampling quadruplet must exclude the neutral"
    N = len(expr_labels)
    matches, diffs = get_matches_and_diffs(expr_labels, ref_labels)
    adjacents, nonadjacents = get_adjacent_and_non_adjacent(expr_labels, ref_labels)
    diffs = diffs * nonadjacents
    diffs = diffs.unsqueeze(1).unsqueeze(1) # N, 1, 1, N
    matches = matches.unsqueeze(-1).unsqueeze(-1) # N, N, 1, 1
    adjacents = adjacents.unsqueeze(1).unsqueeze(-1) # N, 1, N, 1
    quadruplets = matches * adjacents *diffs

    return torch.where(quadruplets)

