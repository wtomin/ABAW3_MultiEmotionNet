import torch
import torchvision
def inception_v3(pretrained=True, remove_classifier=False):
    CNN = torchvision.models.inception_v3(pretrained=pretrained)
    if remove_classifier:
    	layers_to_keep = ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
    	'maxpool1', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'maxpool2', 
    	'Mixed_5b','Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 
    	'Mixed_6d', 'Mixed_6e']
    	layers_to_keep = [getattr(CNN, name) for name in layers_to_keep]
    	CNN = torch.nn.Sequential(*layers_to_keep)
    return CNN

def get_backbone_from_name(name, pretrained=True, remove_classifier=True):
	if name == 'inception_v3':
		backbone =  inception_v3(pretrained=pretrained, remove_classifier=remove_classifier)
		if remove_classifier:
			setattr(backbone, 'features_dim', 768)
			setattr(backbone, 'features_width', 17)
		else:
			setattr(backbone, 'features_dim', 2048)
			# remove the AuxLogits
			if backbone.AuxLogits is not None:
				backbone.AuxLogits = None
		return backbone