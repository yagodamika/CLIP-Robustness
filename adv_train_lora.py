import sys
import os
import CLIP.clip as clip
import torch
import argparse
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pylab as plt
from madgrad import MADGRAD
import random
from torchvision.transforms.functional import to_tensor
import kornia.augmentation as K
import kornia.losses as KL
# import FT_CLIP.FT_CLIP.core as core
from itertools import islice
import numpy as np

# sys.path.append("/home/mika/project/FT_CLIP/FT_CLIP/core")
# from core import *
sys.path.append("/home/mika/project/FT_CLIP1/FT_CLIP2")
# from FT_CLIP1.FT_CLIP2.core import FT_CLIP
from FT_CLIP1.FT_CLIP2.core import FT_CLIP
from FT_CLIP1.FT_CLIP2.lib.CLIP.clip import loralib

cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)


def create_arg_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--adv_p', type=float, default=100,
						help='coefficient for the cosine similarity in the optmization of delta ')
	parser.add_argument('--norm', type=str, default="cosine", help='type of norm to use between embeddings')
	parser.add_argument('--adv_example_iters', type=int, default=10,
						help='number of iterations for the creation of the adversarial example')
	parser.add_argument('--adv_training_iters', type=int, default=100, help='number of iterations')
	parser.add_argument('--lr', type=float, default=0.01, help='learn rate')
	parser.add_argument('--delta_lr', type=float, default=0.01, help='delta learn rate')
	parser.add_argument('--clip_lr', type=float, default=1e-6, help='clip learn rate')  # 1e-9 usually
	parser.add_argument('--clip_lora_lr', type=float, default=1e-5, help='clip learn rate with LoRA')
	parser.add_argument('--alpha', type=float, default=2.0 / 255.0,
						help='coefficient of gradient sign when creating adversarial examples')
	parser.add_argument('--epsilon', type=float, default=8.0 / 255.0, help='maximum perturbation value for PGD')
	parser.add_argument('--batch_size', type=int, default=16, help='batch size')  # 16 # 512
	parser.add_argument('--save_rate', type=int, default=2000, help='save rate')
	parser.add_argument('--dataset_dir', type=str, default='../../../data/dataset/GoogleOpenData/train/data',
						help='dataset directory')
	# Raja5 server path: '../../../data/dataset/GoogleOpenData/train/data'
	# Yoshua server path : '../../../../disk5/datasets/GoogleOpenData/train/data'

	# ARGUMENTS THAT OFTEN NEED TO BE CHANGED
	# TODO: CHANGE
	parser.add_argument('--backbone_name', default='ViT-B/16', type=str,
						help='CLIP model to be used')  # 'ViT-L/14@336px' 'ViT-B/16'
	parser.add_argument('--clip_size', type=int, default=224, help='size of CLIP input images')  # could be 224, 336
	parser.add_argument('--device', type=str, default="cuda:7", help='cuda device to be used')
	parser.add_argument('--res_folder', type=str,
						default="/data/mika/try",
						help='directory of results')
	# "/disk5/mika/clip_experiment/Adversarial_Training/pgd_perm/models/lr_1e6"
	parser.add_argument('--adv_example_method', type=str, default='pgd_perm_targets',
						help='method for getting adversarial examples')  # "l2", "pgd", "pgd_perm_targets"
	parser.add_argument('--seed', type=int, default=0, help='seed')
	parser.add_argument('--lora_r', default=4, type=int,
						help='use any number above 0 to activate LoRA')  # default -1
	parser.add_argument('--lora_alpha', default=1, type=int,
						help='LoRA alpha parameter')  # default 1
	parser.add_argument('--checkpt_start', default=False, type=bool,
						help='Initiliaze trained model from checkpoint')
	parser.add_argument('--cp_start_path',
						default="clip_experiment/Adversarial_Training/pgd_perm_clip_vitL14patches_lora_r8/State_Dict_24000.pth",
						type=str, help='Path to initiliaze trained model from checkpoint')

	parser.add_argument('--starting_epoch', default=0,
						type=int, help='starting epoch if starting from a checkpoint')
	parser.add_argument('--starting_batch_idx', default=0,
						type=int, help='starting batch index if starting from a checkpoint')

	# Contrastive Loss
	parser.add_argument('--symmetric_pos_loss', type=bool, default=False, help='If the positive loss is symmetric')
	parser.add_argument('--pos_loss', type=bool, default=False, help='Use positive loss')

	# original loss
	parser.add_argument('--cosine_sim_loss', type=bool, default=False, help='Use cosine similarity loss')

	# gradient accumulation
	parser.add_argument('--accum_iters', type=int, default=32,
						help='batch accumulation parameter')  # 32 for batch size 16 # 1 is default
	# should be 1 if not doing accumulation

	parser.add_argument('--epochs', type=int, default=1, help='epochs number')  # 5
	parser.add_argument('--dataset', type=str, default="google_opendata", help='dataset')  # google_opendata, imagenet
	return parser.parse_args()


class CropTransform(object):
	def __call__(self, pic):
		random_crop = T.RandomResizedCrop(size=args.clip_size)
		return random_crop(to_tensor(pic))


class ImageDataset(Dataset):
	def __init__(self, root_dir, args, transform=None):
		self.root_dir = root_dir
		self.transform = transform
		self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
		self.args = args

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		# Load image and return it as a tensor
		image = image_load(self.image_paths[idx])
		image = image.to(self.args.device)
		image = image.float()
		image = image / 255.0
		image = image.permute(2, 0, 1)
		if self.transform:
			image = self.transform(image)
		return image


def image_load(image_path):
	with Image.open(image_path) as img:
		# Convert image to RGB if it is not already
		if img.mode != 'RGB':
			img = img.convert('RGB')
		# Convert image to a tensor
		image = torch.from_numpy(np.array(img))
	return image


def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)


def initialize_original_clip(backbone_name, device):
	clip_original, preprocess = clip.load(backbone_name, device="cpu")  # TODO maybe change device

	clip_original.visual = clip_original.visual.to(device)
	clip_original = clip_original.float()
	clip_original.eval().requires_grad_(False)
	return clip_original, preprocess


def initialize_trained_clip(lora_r, lora_alpha, backbone_name, device, checkpt_start, cp_start_path):
	# Initialize CLIP model to be trained

	if lora_r <= 0:  # LoRA is NOT active
		clip_trained = clip.load(backbone_name, device="cpu")[0]

		if checkpt_start:  # If starting from a checkpoint
			state_dict = torch.load(cp_start_path, map_location='cpu')
			clip_trained.load_state_dict(state_dict)

	else:  # LoRA is ACTIVE
		ft_clip = FT_CLIP(backbone_name, lora_r, lora_alpha, device)
		clip_trained = ft_clip.model

		if checkpt_start:  # If starting from a checkpoint
			state_dict = torch.load(cp_start_path, map_location='cpu')
			# TODO: make sure how to load lora dict
			clip_trained.visual.load_state_dict(state_dict, strict=False)

	clip_trained.visual = clip_trained.visual.to(device)
	clip_trained = clip_trained.float()

	return clip_trained

	# TODO move only visual part to device


def get_l2_adv_images(images, clip_trained, clip_normalize, tgt_emd):
	delta = torch.zeros_like(images)
	delta = delta.to(args.device)
	delta.requires_grad = True

	delta_optimizer = torch.optim.Adam([delta], args.delta_lr)

	for i in range(args.adv_example_iters):
		delta_optimizer.zero_grad()

		# Create adversarial example and get its embedding
		adv_example = images + delta
		adv_emb = get_img_embedding(adv_example, clip_trained, clip_normalize)

		# Minimize loss - cosine similarity
		delta_loss = args.adv_p * torch.mean(cosine_similarity(adv_emb, tgt_emd.detach())) + torch.linalg.norm(delta)

		delta_iter_text = f"iteration {i}, delta loss = {delta_loss}\n"
		print(delta_iter_text)

		delta_loss.backward()  # retain_graph=True
		delta_optimizer.step()

	adv_images = images + delta
	delta.requires_grad = False
	return adv_images


def get_pgd_adv_images(images, clip_trained, clip_normalize, tgt_emd):
	adv_images = images.clone().detach()

	for i in range(args.adv_example_iters):
		print(f"pgd iteration: {i}")
		adv_images.requires_grad = True
		adv_emb = get_img_embedding(adv_images, clip_trained, clip_normalize)

		delta_loss = 1 - torch.mean(cosine_similarity(adv_emb, tgt_emd.detach()))
		grad = torch.autograd.grad(delta_loss, adv_images,
								   retain_graph=False, create_graph=False)[0]
		adv_images = adv_images.detach() + args.alpha * grad.sign()
		delta = torch.clamp(adv_images - images, min=-args.epsilon, max=args.epsilon)
		adv_images = torch.clamp(images + delta, min=0, max=1).detach()
	delta.requires_grad = False
	return adv_images


def get_pgd_perm_adv_images(images, clip_trained, clip_normalize, clip_original):
	# get a permutation of the batch images and its embedding
	shuffle_indices = torch.randperm(args.batch_size)
	batch_permutation = images[shuffle_indices]
	perm_emb = get_img_embedding(batch_permutation, clip_original, clip_normalize)

	adv_images = images.clone().detach()

	for i in range(args.adv_example_iters):
		adv_images.requires_grad = True
		adv_emb = get_img_embedding(adv_images, clip_trained, clip_normalize)

		delta_loss = torch.mean(cosine_similarity(adv_emb, perm_emb.detach()))
		grad = torch.autograd.grad(delta_loss, adv_images,
								   retain_graph=False, create_graph=False)[0]
		adv_images = adv_images.detach() + args.alpha * grad.sign()
		delta = torch.clamp(adv_images - images, min=-args.epsilon, max=args.epsilon)
		adv_images = torch.clamp(images + delta, min=0, max=1).detach()
	delta.requires_grad = False
	return adv_images


def get_fgsm_adv_images(images, clip_trained, clip_normalize, tgt_emd):
	adv_images = images.clone().detach()
	adv_images.requires_grad = True
	adv_emb = get_img_embedding(adv_images, clip_trained, clip_normalize)
	delta_loss = 1 - torch.mean(cosine_similarity(adv_emb, tgt_emd.detach()))
	grad = torch.autograd.grad(delta_loss, adv_images, retain_graph=False, create_graph=False)[0]
	adv_images = adv_images.detach() + args.epsilon * grad.sign()
	adv_images = torch.clamp(adv_images, min=0, max=1).detach()
	return adv_images


def get_fgsm_perm_adv_images(images, clip_trained, clip_normalize, clip_original):
	# get a permutation of the batch images and its embedding
	shuffle_indices = torch.randperm(args.batch_size)
	batch_permutation = images[shuffle_indices]
	perm_emb = get_img_embedding(batch_permutation, clip_original, clip_normalize)

	adv_images = images.clone().detach()
	adv_images.requires_grad = True
	adv_emb = get_img_embedding(adv_images, clip_trained, clip_normalize)

	delta_loss = torch.mean(cosine_similarity(adv_emb, perm_emb.detach()))
	grad = torch.autograd.grad(delta_loss, adv_images, retain_graph=False, create_graph=False)[0]
	adv_images = adv_images.detach() + args.epsilon * grad.sign()
	adv_images = torch.clamp(adv_images, min=0, max=1).detach()
	return adv_images


def get_positive_loss(image_features, poss_features, logit_scale, args):
	# calculate S numerator
	logits_per_image_pos = logit_scale * image_features @ poss_features.t()
	ground_truth = (torch.arange(len(logits_per_image_pos)).long()).to(args.device, non_blocking=True)

	if args.symmetric_pos_loss:
		# calculates transposed logits
		logits_per_image_pos_op = logit_scale * poss_features @ image_features.t()

		total_loss = (F.cross_entropy(logits_per_image_pos, ground_truth)
					  + F.cross_entropy(logits_per_image_pos_op, ground_truth)
					  ) / 2
	else:
		total_loss = F.cross_entropy(logits_per_image_pos, ground_truth)

	print(f"calculated positive loss: {total_loss}")

	return total_loss


def train(args):
	iter = 0

	try:
		os.makedirs(args.res_folder)
	except:
		pass

	# Set seed
	set_seed(args.seed)

	# Initialize original CLIP model
	clip_original, preprocess = initialize_original_clip(args.backbone_name, args.device)
	clip_normalize = K.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

	# Initialize CLIP model to be trained
	clip_trained = initialize_trained_clip(args.lora_r, args.lora_alpha, args.backbone_name, args.device,
										   args.checkpt_start, args.cp_start_path)

	# Handle Dataset
	crop = CropTransform()
	random_crop = T.RandomResizedCrop(size=args.clip_size)
	if args.dataset == "google_opendata":
		image_dataset = ImageDataset(args.dataset_dir, args, random_crop)
	if args.dataset == "imagenet":
		image_dataset = datasets.ImageFolder(args.dataset_dir, crop)
		# image_dataset = datasets.ImageNet(args.dataset_dir,split='train',transform=random_crop)
	dataloader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False)

	starting_epoch = 0
	if args.checkpt_start:
		starting_epoch = args.starting_epoch

	for epoch in range(starting_epoch, args.epochs):

		if args.checkpt_start and epoch == starting_epoch:
			# jump to starting batch idx
			loader = islice(dataloader, args.starting_batch_idx)
		else:
			loader = dataloader

		for batch_idx, data in enumerate(loader):

			# jump to starting batch idx
			if args.checkpt_start and epoch == starting_epoch:
				batch_idx += args.starting_batch_idx

			if args.dataset == "imagenet":
				(images, targets) = data
			if args.dataset == "google_opendata":
				images = data

			images = images.to(args.device)

			print(f"epoch # {epoch}, batch # {batch_idx}")
			tgt_emd = get_img_embedding(images, clip_original, clip_normalize)

			######################################################################
			#  Get adversarial examples
			clip_trained.eval().requires_grad_(False)

			if args.adv_example_method == "l2":
				adv_images = get_l2_adv_images(images, clip_trained, clip_normalize, tgt_emd)

			elif args.adv_example_method == "pgd":
				adv_images = get_pgd_adv_images(images, clip_trained, clip_normalize, tgt_emd)

			elif args.adv_example_method == "pgd_perm_targets":
				adv_images = get_pgd_perm_adv_images(images, clip_trained, clip_normalize, clip_original)

			elif args.adv_example_method == "FGSM":
				adv_images = get_fgsm_adv_images(images, clip_trained, clip_normalize, tgt_emd)

			elif args.adv_example_method == "FGSM_perm_targets":
				adv_images = get_fgsm_perm_adv_images(images, clip_trained, clip_normalize, clip_original)

			######################################################################
			# train clip
			clip_trained.visual.train().requires_grad_(True)
			if args.lora_r <= 0:  # LoRA is NOT active
				adv_clip_optimizer = torch.optim.Adam(clip_trained.visual.parameters(), args.clip_lr)

			else:  # LoRA is ACTIVE
				loralib.mark_only_lora_as_trainable(clip_trained.visual)
				trainable_parameters = [p for p in clip_trained.visual.parameters() if p.requires_grad]
				adv_clip_optimizer = torch.optim.Adam(trainable_parameters, args.clip_lora_lr)

			# fine tuning of the model

			adv_emd = clip_trained.encode_image(adv_images)

			loss = 0
			if args.cosine_sim_loss:
				loss = 1 - torch.mean(cosine_similarity(adv_emd, tgt_emd.detach()))
			if args.pos_loss:
				loss = get_positive_loss(tgt_emd.detach(), adv_emd, clip_trained.logit_scale.exp(), args)

			# normalize loss to account for batch accumulation
			loss = loss / args.accum_iters

			# backward pass
			loss.backward()

			# weights update
			if ((batch_idx + 1) % args.accum_iters == 0) or (batch_idx + 1 == len(dataloader)):
				adv_clip_optimizer.step()
				adv_clip_optimizer.zero_grad()

			######################################################################
			# Adversarial Training is done

			# Save state dictionaries
			checkpoint_path = f'{args.res_folder}/State_Dict_{epoch}_{batch_idx}.pth'
			if (batch_idx % args.save_rate) == 0:
				if args.lora_r <= 0:  # LoRA is NOT active
					state_dict = clip_trained.state_dict()
					torch.save(state_dict, checkpoint_path)
				else:  # LoRA is ACTIVE
					state_dict = loralib.lora_state_dict(clip_trained.visual)
					torch.save(state_dict, checkpoint_path)


def get_img_embedding(y, clip_model, clip_normalize):
	y = TF.resize(y, (args.clip_size, args.clip_size))

	# Make picture suitable for CLIP
	input_img = clip_normalize(y)

	# Encode image with CLIP
	embed = clip_model.encode_image(input_img)  # .float()
	return embed


def cosine_similarity(x, y):
	dist = cossim(x, y)
	return dist


# TODO : update function with correct arguments
def create_log_file(args):
	log_file_path = args.res_folder + "/Log.txt"
	log_file = open(log_file_path, "a")

	text = ""

	text += f"coefficient for the cosine similarity in the optimization of delta: {args.adv_p}\n"

	text += f"Used similarity loss: {args.norm}\n"

	text += f"Delta Iterations number: {args.adv_example_iters}\n"
	text += f"Adversarial Training Iterations number: {args.adv_training_iters}\n"

	text += f"Optimizer type: {args.optim_type}\n"
	text += f"Learn rate: {args.lr}\n"
	text += f"Delta Learn rate: {args.delta_lr}\n"
	text += f"CLIP Learn rate: {args.clip_lr}\n"

	text += f"Seed: {args.seed}\n"
	text += f"Display Rate: {args.display_rate}\n"

	log_file.write(text)
	log_file.close()


args = create_arg_parser()
train(args)
