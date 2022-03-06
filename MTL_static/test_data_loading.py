from data.ABAW3_DataModule import get_Dataset_TrainVal, MTL_DataModule
from utils.data_utils import train_transforms, test_transforms

def load_all_datasets_to_datamodule(video, batch_size, seq_len=None):
	dataset1 = get_Dataset_TrainVal("create_annotation_file/AU/AU_annotations.pkl", video=video,
		transforms_train=train_transforms(112), transforms_test=test_transforms(112), 
		seq_len=seq_len)
	dataset2 = get_Dataset_TrainVal("create_annotation_file/EXPR/EXPR_annotations.pkl", video=video,
		transforms_train=train_transforms(112), transforms_test=test_transforms(112), 
		seq_len=seq_len)
	dataset3 = get_Dataset_TrainVal("create_annotation_file/VA/VA_annotations.pkl", video=video,
		transforms_train=train_transforms(112), transforms_test=test_transforms(112), 
		seq_len=seq_len)

	dataset4 = get_Dataset_TrainVal("create_annotation_file/MTL/AU_VA_annotations.pkl", video=video,
		transforms_train=train_transforms(112), transforms_test=test_transforms(112), 
		seq_len=seq_len)
	dataset5 = get_Dataset_TrainVal("create_annotation_file/MTL/AU_EXPR_VA_annotations.pkl", video=video,
		transforms_train=train_transforms(112), transforms_test=test_transforms(112), 
		seq_len=seq_len)
	dm = MTL_DataModule(video,
		[dataset1[0], dataset2[0], dataset3[0]], 
		[dataset1[1], dataset2[1], dataset3[1]], batch_size,
		[dataset4[0], dataset5[0]], [dataset4[1], dataset5[1]],
		max(1, batch_size*0.03),
		num_workers_train = 0, num_workers_test=0)
	train_dataloaders = dm.train_dataloader()
	val_dataloaders = dm.val_dataloader()

	for key in ['single', 'multiple']:
		dataloader = train_dataloaders[key]
		dataloader = iter(dataloader)
		batch = next(dataloader)
		if key =='single':
			(x_au, y_au), (x_expr, y_expr), (x_va, y_va) = batch
		else:
			(x_au_va, y_au_va), (x_au_expr_va, y_au_expr_va) = batch


if __name__=="__main__":
	# Load image datasets
	# video = False
	# batch_size = 24
	# load_all_datasets_to_datamodule(video, batch_size)

	video = True
	batch_size = 3
	seq_len = 32
	load_all_datasets_to_datamodule(video, batch_size, seq_len)