from openke.config import Trainer, Tester
from openke.module.modelIdx import LN_TransE
from openke.data import TrainDataLoader, TestDataLoader
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=3, help='cuda idx.')
parser.add_argument('--dim', type=int, default=4, help='embedding dim.')
parser.add_argument('--odim', type=int, default=512, help='output embedding dim.')
parser.add_argument('--dt', type=str, default='WN18RR', help='WN18RR,FB15K237,YAGO3-10.')
parser.add_argument('--l', type=int, default=2, help='1,2,3,4.')
args = parser.parse_args()

torch.manual_seed(12)

dim0 = args.dim
dim1 = args.odim
cuda_ = "cuda:"+str(args.cuda)
dt = args.dt
layers = args.l

hidden_dim = [200, 800, 800, 800]
dim = [dim0,dim1]

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/"+dt+"/",
	nbatches = 10,
	threads = 1, 
	sampling_mode = "normal",
	bern_flag = 1, 
	filter_flag = 1, 
    neg_ent = 0,
	neg_rel = 0,
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/"+dt+"/", "link")

# define the model
transe = LN_TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	hidden = hidden_dim,
	layers = layers,
	dim = dim, 
	p_norm = 1,
	norm_flag = True,
	pos_para = 1,
	neg_para = 0.0001,
)

max_mrr = 0
max_hit = [0,0,0]

trainer = Trainer(model = transe, data_loader = train_dataloader, train_times = 10, alpha = 0.0001, use_gpu = True, cuda_idx = cuda_, on_step=True, opt_method ="adam")
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True, cuda_idx = cuda_, trainer = trainer)

for idx in range(1, 500):
	print("		#####", idx)
	# train the model
	trainer.run()
	transe.save_checkpoint('./'+dt+'/transe_idx'+str(dim)+'.ckpt')

# test the model
transe.load_checkpoint('./'+dt+'/transe_idx'+str(dim)+'.ckpt')
mrr, mr, hit10, hit3, hit1 = tester.run_link_prediction(type_constrain = False)

