import warnings
warnings.filterwarnings('ignore')

import os
# import keras
import numpy as np
from utils import *
from tqdm import *
# from evaluate import evaluate
from eval import evaluateTorch
import tensorflow as tf
# import keras.backend as K
# from keras.layers import *
from models import *
import torch.nn.functional as F
import torch
from models_encodercat import Encoder_Model
from models_encodercatal import Encoder_Model_Al
import random


def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True  
#sess = tf.Session(config=config)  

seed =12306 
np.random.seed(seed)

set_seed(seed)
tf.compat.v1.set_random_seed(seed)


data_ca="ja_en/"
# train_pair,dev_pair,adj_matrix,r_index,r_val,adj_features,rel_features = load_data("data/fb_yago_en/",train_ratio=0.20)
data_dir = 'data/'+data_ca
_,_,_,adj_matrix,r_index,r_val,adj_features,rel_features = load_data(data_dir,train_ratio=0.30)

output_dir="checkpoints/"+data_ca

ref_pairs = np.load(os.path.join(output_dir, "ref_pairs.npz"))
train_pair = ref_pairs["train_pair"]
val_pair = ref_pairs["val_pair"]
dev_pair = ref_pairs["dev_pair"]

used_inds3_array = np.load(os.path.join(output_dir, "used_inds3.npz"))
used_inds3_array=used_inds3_array['used_inds3']
used_inds=used_inds3_array.tolist()

visual_links_array3 = np.load(os.path.join(output_dir, "visual_links_array3.npz"))
visual_links_array3 = visual_links_array3['visual_links_array3']
visual_links = visual_links_array3.tolist()


adj_matrix = np.stack(adj_matrix.nonzero(),axis = 1)
rel_matrix,rel_val = np.stack(rel_features.nonzero(),axis = 1),rel_features.data
ent_matrix,ent_val = np.stack(adj_features.nonzero(),axis = 1),adj_features.data




node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
triple_size = len(adj_matrix)
node_hidden = 128
rel_hidden = 128
batch_size = 1024
dropout_rate = 0.5
lr = 0.005
gamma = 1
depth = 2

device = "cuda:0"






# img_features = load_img_features(node_size, "data/fb_dbk_en/FB15K_DB15K")
# img_features = load_img_features(node_size, "data/fb_dbk_en/FB15K_YAGO15K")
img_features = load_img_features(node_size, data_dir)
img_features = F.normalize(torch.Tensor(img_features).to(device))
print("image feature shape:", img_features.shape)


triple_size = len(adj_matrix)  # not triple size, but number of diff(h, t)
eval_epoch = 3



training_time = 0.
grid_search_time = 0.
time_encode_time = 0.

 
e1 = os.path.join(data_dir,'ent_ids_1')
e2 = os.path.join(data_dir,'ent_ids_2')
left_ents = get_ids(e1)
right_ents = get_ids(e2)

unsup=True
unsup_k=6000



# visual_links=[]
# used_inds = []

l_img_f = img_features[left_ents] # left images
r_img_f = img_features[right_ents] # right images

img_sim = l_img_f.mm(r_img_f.t())
topk =unsup_k
two_d_indices,vals = get_topk_indices(img_sim, topk*100)
del l_img_f, r_img_f, #img_sim

l_img_dev = img_features[dev_pair[:,0]]
r_img_dev = img_features[dev_pair[:,1]]
img_sim_dev = l_img_dev.mm(r_img_dev.t())

l_img_val = img_features[val_pair[:,0]]
r_img_val = img_features[val_pair[:,1]]
img_sim_val = l_img_val.mm(r_img_val.t())

# count = 0
# for index in range(len(two_d_indices)):
#     ind = two_d_indices[index]
#     if left_ents[ind[0]] in used_inds: continue
#     if right_ents[ind[1]] in used_inds: continue
#     used_inds.append(left_ents[ind[0]])
#     used_inds.append(right_ents[ind[1]])
#     visual_links.append((left_ents[ind[0]], right_ents[ind[1]],vals[index].cpu(),vals[index].cpu()))
#     # visual_links.append((left_ents[ind[0]], right_ents[ind[1]]))

#     count += 1
#     if count == topk: 
#         break


# num_al=1500
# count_al=0
# for index in range(len(train_pair)):
#     ind = train_pair[index]
#     if ind[0] in used_inds: continue
#     if ind[1] in used_inds: continue
#     visual_links.append((ind[0], ind[1],1,1))
#     count_al +=1
#     if count_al==num_al:
#         break


# for index in range(1500):
#     ind = train_pair[index]
#     visual_links.append((ind[0], ind[1],1,1))

train_pair_base=train_pair


print('begin')
# inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
# model = OverAll(node_size=node_size, node_hidden=node_hidden,
#                 rel_size=rel_size, rel_hidden=rel_hidden,
#                 rel_matrix=rel_matrix, ent_matrix=ent_matrix,
#                 triple_size=triple_size, dropout_rate=dropout_rate,
#                 depth=depth, device=device)
# model = model.to(device)

adj_matrix = torch.from_numpy(np.transpose(adj_matrix))
rel_matrix = torch.from_numpy(np.transpose(rel_matrix))
ent_matrix = torch.from_numpy(np.transpose(ent_matrix))
r_index = torch.from_numpy(np.transpose(r_index))
r_val = torch.from_numpy(r_val)


entity_label = np.zeros((node_size,node_hidden))
for item in visual_links:
    entity_label[item[0]]=np.array([1]*node_hidden)
    entity_label[item[1]]=np.array([1]*node_hidden)

model_al = Encoder_Model_Al(node_size=node_size, node_hidden=node_hidden,
                rel_size=rel_size, rel_hidden=rel_hidden,new_node_size=node_size,
                rel_matrix=rel_matrix, ent_matrix=torch.tensor(entity_label),r_index=r_index,r_val=r_val,
                adj_matrix=adj_matrix,img_feature = None,
                triple_size=triple_size, dropout_rate=dropout_rate,
                depth=depth, device=device)
model_al = model_al.to(device)




# if os.path.exists(os.path.join(output_dir, "check_model3.tar")):
#     # self.model.load_weights(os.path.join(from_dir, "model.ckpt"))
#     pretrained_dict = torch.load(os.path.join(output_dir, "check_model3.tar"))['state_dict']
#     model_dict = model_al.state_dict()

#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in {"rel_embedding.weight", "e_encoder.attn_kernels.0","e_encoder.attn_kernels.1"}}
#     print(pretrained_dict.keys())
#     model_dict.update(pretrained_dict)
#     model_al.load_state_dict(model_dict)
# else:
#     print("can not load previouse model")

model_al.eval()
with torch.no_grad():
     train_paire_al = model_al(train_pair_base).mean(-1)

sorted, indices = torch.sort(train_paire_al) 

selected_train=train_pair[indices.cpu()]

num_al=500
count_al=0
for index in range(len(selected_train)):
    ind = selected_train[index]
    if ind[0] in used_inds: continue
    if ind[1] in used_inds: continue
    used_inds.append(ind[0])
    used_inds.append(ind[1])
    visual_links.append((ind[0], ind[1],1,1))
    count_al +=1
    if count_al==num_al:
        break

train_pair = np.array(visual_links, dtype=np.float32)




model = Encoder_Model(node_size=node_size, node_hidden=node_hidden,
                rel_size=rel_size, rel_hidden=rel_hidden,new_node_size=node_size,
                rel_matrix=rel_matrix, ent_matrix=torch.tensor(ent_matrix),r_index=r_index,r_val=r_val,
                adj_matrix=adj_matrix,img_feature = img_features,
                triple_size=triple_size, dropout_rate=dropout_rate,
                depth=depth, device=device)
model = model.to(device)
# opt = torch.optim.RMSprop(model.parameters(), lr=lr)
opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.0005)
print('model constructed')



# evaluater = evaluate(dev_pair)
evaluater_torch = evaluateTorch(device,dev_pair)
evaluater_torch_val = evaluateTorch(device,val_pair)
rest_set_1 = [e1 for e1, e2 in dev_pair]
rest_set_2 = [e2 for e1, e2 in dev_pair]
np.random.shuffle(rest_set_1)
np.random.shuffle(rest_set_2)

model.eval()
with torch.no_grad():
    # dev_pair = torch.from_numpy(dev_pair).to(device)
    Lvec, Rvec = model.get_embeddings(dev_pair[:, 0], dev_pair[:, 1],turn=0)
    # output = model(inputs)
    # Lvec, Rvec = get_embedding(dev_pair[:, 0], dev_pair[:, 1], output.cpu())
    # evaluater.test(Lvec, Rvec)
    evaluater_torch.test(Lvec, Rvec,img_sim_dev.cpu(),img_sim_dev.t().cpu())
epoch = 12
mrr_or=0
for turn in range(5):

    for i in trange(epoch):
        model.train()
        np.random.shuffle(train_pair)
        for pairs in [train_pair[i * batch_size:(i + 1) * batch_size] for i in
                        range(len(train_pair) // batch_size + 1)]:
            inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix]
            # output = model(inputs)
            # loss = align_loss(pairs, output)
            pairs = torch.from_numpy(pairs).to(device)
            loss  =model(pairs,turn)
            # print(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # if i == epoch - 1:
            # toc = time.time()
        model.eval()
        with torch.no_grad():
            Lvec, Rvec = model.get_embeddings(dev_pair[:, 0], dev_pair[:, 1],turn)
            Lvec_val, Rvec_val = model.get_embeddings(val_pair[:, 0], val_pair[:, 1],turn)

            if turn >=4:
                mrr=evaluater_torch_val.test(Lvec_val, Rvec_val,img_sim_val.cpu(),img_sim_val.t().cpu())

                if mrr>=mrr_or:
                    mrr_or =mrr
                    mrr = evaluater_torch.test(Lvec, Rvec,img_sim_dev.cpu(),img_sim_dev.t().cpu())
                    torch.save({'state_dict': model.state_dict()}, os.path.join(output_dir, "check_model3.tar"))
                    visual_links_array = np.array(visual_links, dtype=np.int32)
                    np.savez(os.path.join(output_dir, "visual_links_array3.npz"), visual_links_array3=visual_links_array)
                    used_inds_array = np.array(used_inds, dtype=np.int32)
                    np.savez(os.path.join(output_dir, "used_inds3.npz"), used_inds3=used_inds_array)
  

        new_pair = []

    l_img_dev_boot = img_features[rest_set_1]
    r_img_dev_boot = img_features[rest_set_2]
    img_sim_dev_boot = l_img_dev_boot.mm(r_img_dev_boot.t())

    Lvec,Rvec = model.get_embeddings(rest_set_1,rest_set_2,turn)
    # A,B,A_score,B_score = evaluater.CSLS_cal(Lvec.detach().numpy(),Rvec.detach().numpy(),False)
    A,B = evaluater_torch.CSLS_cal(Lvec.detach(),Rvec.detach(),img_sim_dev_boot.cpu(),img_sim_dev_boot.t().cpu(),False)

    
    for i,j in enumerate(A):
        if  B[j] == i:

            new_pair.append([rest_set_1[j],rest_set_2[i],1,1])

    A_score_sorted = np.sort(np.array(new_pair)[:,2])


    B_score_sorted = np.sort(np.array(new_pair)[:,3])




    train_pair = np.concatenate([train_pair,np.array(new_pair)],axis = 0)

    for e1,e2,_,_ in new_pair:
        if e1 in rest_set_1:
            rest_set_1.remove(e1) 
        
    for e1,e2,_,_ in new_pair:
        if e2 in rest_set_2:
            rest_set_2.remove(e2)
    epoch = 5
    

# torch.save({'state_dict': model.state_dict()}, os.path.join(output_dir, "check_model4.tar"))
# visual_link_array = np.array(visual_links, dtype=np.int32)
# np.savez(os.path.join(output_dir, "visual_links_array4.npz"), visual_links_array4=visual_link_array)
# used_inds_array = np.array(used_inds,  dtype=np.int32)
# np.savez(os.path.join(output_dir, "used_inds4.npz"), used_inds4=used_inds_array)






