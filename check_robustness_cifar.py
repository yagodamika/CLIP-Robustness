import sys
import os

sys.path.append('./deep_image_prior')
from deep_image_prior.models import *
from deep_image_prior.utils.sr_utils import *
import CLIP.clip as clip
import torch
import argparse
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.nn import functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pylab as plt
from madgrad import MADGRAD
import random
import kornia.augmentation as K
import kornia.losses as KL
from torchvision.transforms._presets import ImageClassification
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR100
from FT_CLIP1.FT_CLIP2.core import FT_CLIP

device = "cuda:7" if torch.cuda.is_available() else "cpu"
cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

def cosine_similarity(x, y):
	dist = cossim(x, y)
	return dist


def get_img_embedding(y, clip_model, clip_normalize):
	input_img = clip_normalize(y)
	embed = clip_model.encode_image(input_img)
	return embed


def get_txt_embedding(txt, clip_model):
	"""
	tokenizer = clip.tokenize
	# tokenize the texts
	#tokens = [tokenizer(text) for text in txt_list]
	# encode the tokenized texts
	#embeddings = clip_model.encode_text(torch.stack(tokens).to(device)).float()
	embeddings = clip_model.encode_text(txt_list).float()
	"""

	embed = clip_model.encode_text(clip.tokenize(txt).to(device)).float()

	return embed

def pgd(images, clip_original, clip_normalize, args, tgt_emd):
	adv_images = images.clone().detach()

	for j in range(args.adv_example_iters):
		adv_images.requires_grad = True
		adv_emb = get_img_embedding(adv_images, clip_original, clip_normalize)

		delta_loss = 1 - torch.mean(cosine_similarity(adv_emb, tgt_emd.detach()))
		grad = torch.autograd.grad(delta_loss, adv_images,
								   retain_graph=False, create_graph=False)[0]
		adv_images = adv_images.detach() + args.alpha * grad.sign()
		delta = torch.clamp(adv_images - images, min=-args.epsilon, max=args.epsilon)
		adv_images = torch.clamp(images + delta, min=0, max=1).detach()

	return adv_images

def pgd_perm_targets(images, clip_original, clip_normalize, args):
	# get a permutation of the batch images and its embedding
	shuffle_indices = torch.randperm(args.batch_size)
	batch_permutation = images[shuffle_indices]
	perm_emb = get_img_embedding(batch_permutation, clip_original, clip_normalize)

	adv_images = images.clone().detach()

	for j in range(args.adv_example_iters):
		adv_images.requires_grad = True
		adv_emb = get_img_embedding(adv_images, clip_original, clip_normalize)

		delta_loss = torch.mean(cosine_similarity(adv_emb, perm_emb.detach()))
		grad = torch.autograd.grad(delta_loss, adv_images,
								   retain_graph=False, create_graph=False)[0]
		adv_images = adv_images.detach() + args.alpha * grad.sign()
		delta = torch.clamp(adv_images - images, min=-args.epsilon, max=args.epsilon)
		adv_images = torch.clamp(images + delta, min=0, max=1).detach()

	return adv_images

def initialize_trained_clip(lora_r, lora_alpha, backbone_name, device, checkpt_start, cp_start_path, tecoa):
	# Initialize CLIP model to be trained

	if tecoa:
		clip_trained, _ = clip.load(backbone_name, device=device)
		state_dict = torch.load(cp_start_path, map_location='cpu')
		clip_dict = state_dict["vision_encoder_state_dict"]  # "CLIP_dict"
		clip_trained.visual.load_state_dict(clip_dict)

	elif lora_r <= 0:  # LoRA is NOT active
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

def check_robustness(args):
	global lora_r, lora_alpha

	# Set seed
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	random.seed(args.seed)

	# open log file
	log_file_path = args.res_folder + "/Log.txt"
	log_file = open(log_file_path, "a")

	# Initialize original CLIP model
	clip_original, preprocess = clip.load(args.backbone_name, device=device)
	clip_original = clip_original.to(device)
	clip_original = clip_original.float()
	clip_original.eval().requires_grad_(False)
	clip_size = args.clip_size

	# Initialize trained CLIP model
	clip_trained = initialize_trained_clip(args.lora_r, args.lora_alpha, args.backbone_name, device, True, args.trained_path, args.tecoa)
	clip_normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
	clip_trained.eval().requires_grad_(False)
	clip_trained.to(device)

	# Download the dataset
	cifar100 = CIFAR100(root=args.cifar_path, download=True, train=False)
	orig_cs_sum = 0.0  # Original CLIP cosine similarity sum
	trained_cs_sum = 0.0  # Trained CLIP cosine similarity sum
	success_number = 0.0

	for i in range(len(cifar100)):

		# if i == args.num_iterations:
		#    break

		images, class_id = cifar100[i]

		images = preprocess(images).unsqueeze(0).to(device)

		text_label = "a photo of a " + cifar100.classes[class_id]

		tgt_emd = get_txt_embedding(text_label, clip_original)

		if args.adv_example_method == "pgd":
			adv_images = pgd(images, clip_original, clip_normalize, args, tgt_emd)

		if args.adv_example_method == "pgd_perm_targets":
			adv_images = pgd_perm_targets(images, clip_original, clip_normalize, args)

		# Compare cosine similarity between adversarial
		# examples in original CLIP space and trained CLIP space
		print(f"Adversarial images created, number {i}")

		# Original CLIP
		original_txt_emb = tgt_emd
		original_adv_emb = get_img_embedding(adv_images, clip_original, clip_normalize)
		original_cosine_similarity = cosine_similarity(original_txt_emb, original_adv_emb)

		# Trained CLIP
		trained_txt_emb = get_txt_embedding(text_label, clip_trained)
		trained_adv_emb = get_img_embedding(adv_images, clip_trained, clip_normalize)
		trained_cosine_similarity = cosine_similarity(trained_txt_emb, trained_adv_emb)

		if trained_cosine_similarity >= original_cosine_similarity:
			success_number += 1.0

		# write result to file
		text = f"Batch {i}: Original cosine similarity: {original_cosine_similarity.mean()} , Trained: {trained_cosine_similarity.mean()}\n"
		log_file.write(text)

		orig_cs_sum += original_cosine_similarity
		trained_cs_sum += trained_cosine_similarity

		if i % args.save_rate == 0:
			result = TF.to_pil_image(adv_images[0])
			result.save(f'{args.res_folder}/out_{i}.png', quality=100)
			if i != 0:
				# Calculate average cosine similarity
				orig_cs_mean = orig_cs_sum.item() / float(i)
				trained_cs_mean = trained_cs_sum.item() / float(i)

				success_rate = success_number / float(i)

				text = f"/n iteration {i}:\nOriginal cosine similarity average: {orig_cs_mean} , Trained: {trained_cs_mean}\n"
				log_file.write(text)
				text = f"Success rate: {success_rate:.4f}\n"
				log_file.write(text)

	# Calculate average cosine similarity
	orig_cs_mean = orig_cs_sum.item() / float(i)
	trained_cs_mean = trained_cs_sum.item() / float(i)

	success_rate = success_number / float(i)

	text = f"Original cosine similarity average: {orig_cs_mean} , Trained: {trained_cs_mean}\n"
	log_file.write(text)
	text = f"Success rate: {success_rate:.4f}"
	log_file.write(text)

	log_file.close()


# TODO : update function with correct arguments
def create_log_file(args):
	log_file_path = args.res_folder + "/Log.txt"
	log_file = open(log_file_path, "a")

	text = ""
	text += f"Delta Iterations number: {args.adv_example_iters}\n"
	text += f"Optimizer type: {args.optim_type}\n"
	text += f"Seed: {args.seed}\n"
	text += f"Display Rate: {args.display_rate}\n"

	log_file.write(text)
	log_file.close()


# TODO : update function with correct arguments
def create_arg_parser():
	parser = argparse.ArgumentParser()

	#  SAVING
	parser.add_argument('--res_folder', type=str,
						default="/data/mika/clip_experiment/SDS_training/tecoa/checks/robustness",
						help='directory of results')
	parser.add_argument('--save_rate', type=int, default=1000, help='save rate')

	#	CLIP MODEL
	parser.add_argument('--backbone_name', type=str, default="ViT-B/32",
						help='backbone model name')  # 'ViT-L/14@336px' 'ViT-B/16'
	parser.add_argument('--tecoa', type=bool, default=True, help="Whether to use TeCoA CLIP")
	parser.add_argument('--lora_r', type=int, default=4, help="lora r value")
	parser.add_argument('--lora_alpha', type=int, default=1, help="lora alpha value")
	parser.add_argument('--trained_path', type=str,
						default="/data/mika/TeCoAmodel_best.pth.tar",
						help="trained model's path")
	# Tecoa raja5 path: "/data/mika/TeCoAmodel_best.pth.tar"
	parser.add_argument('--clip_size', type=int, default=224, help="size of CLIP's input")  # 224, 336

	#   DATASET
	parser.add_argument('--cifar_path', type=str, default="/data/dataset/cifar100", help="CIFAR100 path")
	# yoshua path "../../../../disk5/datasets/cifar100"
	# Raja5 path "/data/dataset/cifar100"

	#  ADVERSARIAL EXAMPLES
	parser.add_argument('--adv_example_method', type=str, default='pgd',
						help='method for getting adversarial examples')  # "l2", "pgd", "pgd_perm_targets"
	parser.add_argument('--alpha', type=float, default=2.0 / 255.0,
						help='coefficient of gradient sign when creating adversarial examples')
	parser.add_argument('--epsilon', type=float, default=8.0 / 255.0, help='maximum perturbation value for PGD')
	parser.add_argument('--adv_example_iters', type=int, default=10,
						help='number of iterations for the creation	of the adversarial example')

	#  OPTIMIZATION PARAMS
	parser.add_argument('--num_iterations', type=int, default=1000000, help='save rate')
	parser.add_argument('--seed', type=int, default=0, help='seed')
	parser.add_argument('--batch_size', type=int, default=16, help='batch size')

	return parser.parse_args()

args = create_arg_parser()
check_robustness(args)
