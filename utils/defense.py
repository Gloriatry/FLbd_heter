# -*- coding = utf-8 -*-
import numpy as np
import torch
import copy
import time
import hdbscan
import math
import sklearn.metrics.pairwise as smp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def cos(a, b):
    # res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-
    res = (np.dot(a, b) + 1e-9) / (np.linalg.norm(a) + 1e-9) / \
        (np.linalg.norm(b) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res


def fltrust(params, central_param, global_parameters, args):
    FLTrustTotalScore = 0
    score_list = []
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    sum_parameters = None
    for local_parameters in params:
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)
        # 计算cos相似度得分和向量长度裁剪值
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)
        client_clipped_value = central_norm/torch.norm(local_parameters_v)
        score_list.append(client_cos)
        FLTrustTotalScore += client_cos
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in local_parameters.items():
                # 乘得分 再乘裁剪值
                sum_parameters[key] = client_cos * \
                    client_clipped_value * var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + client_cos * client_clipped_value * local_parameters[
                    var]
    if FLTrustTotalScore == 0:
        print(score_list)
        return global_parameters
    for var in global_parameters:
        # 除以所以客户端的信任得分总和
        temp = (sum_parameters[var] / FLTrustTotalScore)
        if global_parameters[var].type() != temp.type():
            temp = temp.type(global_parameters[var].type())
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
        else:
            global_parameters[var] += temp * args.server_lr
    print(score_list)
    return global_parameters


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector_flt_cpu(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        vec.append(param.cpu().view(-1))
    return torch.cat(vec)


def no_defence_balance(params, global_parameters):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone()
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += (sum_parameters[var] / total_num)

    return global_parameters


def multi_krum(gradients, n_attackers, args, writer, epoch, multi_k=False):

    grads = flatten_grads(gradients)

    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))

    while len(remaining_updates) > 2 * n_attackers + 2:
        # torch.cuda.empty_cache()
        distances = []
        scores = None
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        
        values_dict = {f'Client_{i}': value for i, value in enumerate(scores)}
        writer.add_scalars('krum_distances', values_dict, epoch)        

        print(scores)
        args.krum_distance.append(scores)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break

    # aggregate = torch.mean(candidates, dim=0)

    # return aggregate, np.array(candidate_indices)
    num_clients = max(int(args.frac * args.num_users), 1)
    num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients
    args.turn+=1
    if multi_k == False:
        if candidate_indices[0] < num_malicious_clients:
            args.wrong_mal += 1
            
    print(candidate_indices)
    
    print('Proportion of malicious are selected:'+str(args.wrong_mal/args.turn))

    for i in range(len(scores)):
        if i < num_malicious_clients:
            args.mal_score += scores[i]
        else:
            args.ben_score += scores[i]
    
    # return np.array(candidate_indices)
    return candidate_indices



def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend(
                    [grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs




def get_update(update, model):
    '''get the update weight'''
    update2 = {}
    for key, var in update.items():
        update2[key] = update[key] - model[key]
    return update2



def RLR(global_model, agent_updates_list, args, writer, epoch):
    """
    agent_updates_dict: dict['key']=one_dimension_update
    agent_updates_list: list[0] = model.dict
    global_model: net
    """
    # args.robustLR_threshold = 6
    args.server_lr = 1

    grad_list = []
    for i in agent_updates_list:
        grad_list.append(parameters_dict_to_vector_rlr(i))
    agent_updates_list = grad_list
    

    aggregated_updates = 0
    for update in agent_updates_list:
        # print(update.shape)  # torch.Size([1199882])
        aggregated_updates += update
    aggregated_updates /= len(agent_updates_list)
    lr_vector = compute_robustLR(agent_updates_list, args)
    writer.add_scalar('RLR_rate', (lr_vector<0).sum().item()/lr_vector.numel(), epoch)
    cur_global_params = parameters_dict_to_vector_rlr(global_model.state_dict())
    new_global_params =  (cur_global_params + lr_vector*aggregated_updates).float() 
    global_w = vector_to_parameters_dict(new_global_params, global_model.state_dict())
    # print(cur_global_params == vector_to_parameters_dict(new_global_params, global_model.state_dict()))
    return global_w

def parameters_dict_to_vector_rlr(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        vec.append(param.view(-1))
    return torch.cat(vec)

def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)



def vector_to_parameters_dict(vec: torch.Tensor, net_dict) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """

    pointer = 0
    for param in net_dict.values():
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
    return net_dict

def compute_robustLR(params, args):
    agent_updates_sign = [torch.sign(update) for update in params]  
    sm_of_signs = torch.abs(sum(agent_updates_sign))
    # print(len(agent_updates_sign)) #10
    # print(agent_updates_sign[0].shape) #torch.Size([1199882])
    sm_of_signs[sm_of_signs < args.robustLR_threshold] = -args.server_lr
    sm_of_signs[sm_of_signs >= args.robustLR_threshold] = args.server_lr 
    return sm_of_signs.to(args.gpu)
   
    


def flame(local_model, update_params, global_model, args, norm_writer, epoch, excluded_frequency):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list=[]
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1- cos(local_model_vector[i],local_model_vector[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)
    num_clients = 10
    # num_clients = max(int(args.frac * args.num_users), 1)
    # num_malicious_clients = int(args.malicious * num_clients)
    # num_benign_clients = num_clients - num_malicious_clients
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1,min_samples=1,allow_single_cluster=True).fit(cos_list)
    print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster=0
    max_cluster_index=0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max()+1):
            if len(clusterer.labels_[clusterer.labels_==index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_==index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
        for i in range(len(local_model_vector)):
            # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
            norm_list = np.append(norm_list,torch.norm(parameters_dict_to_vector(update_params[i]),p=2).item())  # no consider BN
    print(benign_client)
    
    print(excluded_frequency)
    for i in range(args.num_users):
        if i not in benign_client:
            excluded_frequency[i] += 1

    # for i in range(len(benign_client)):
    #     if benign_client[i] < num_malicious_clients:
    #         args.wrong_mal+=1
    #     else:
    #         #  minus per benign in cluster
    #         args.right_ben += 1
    # args.turn+=1
    # if math.isclose(args.malicious, 0) == False:
    #     print('proportion of malicious are selected:',args.wrong_mal/(num_malicious_clients*args.turn))
    # print('proportion of benign are selected:',args.right_ben/(num_benign_clients*args.turn))
    
    clip_value = np.median(norm_list)
    gama_list = []
    for i in range(len(norm_list)):
        gama_list.append(clip_value/norm_list[i])
    norm_values_dict = {f'Client_{i}': value for i, value in enumerate(gama_list)}
    norm_writer.add_scalars('flame_norm', norm_values_dict, epoch)

    # clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value/norm_list[benign_client[i]]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model = no_defence_balance([update_params[i] for i in benign_client], global_model)
    #add noise
    for key, var in global_model.items():
        if key.split('.')[-1] == 'num_batches_tracked':
                    continue
        temp = copy.deepcopy(var)
        temp = temp.normal_(mean=0,std=args.noise*clip_value)
        var += temp
    return global_model

def get_pca(data):
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    return data

def average_weights(w, marks):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * marks[0]
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * marks[i]
        w_avg[key] = w_avg[key] *(1/sum(marks))
    return w_avg

def fl_defender(global_model, local_models, score_history, selected_users, filename, epoch):
    global_model = list(global_model.parameters())
    last_g = global_model[-2].cpu().data.numpy()  # 返回的是最后一层线性层的权重
    m = len(local_models)
    f_grads = [None for i in range(m)]
    for i in range(m):
        grad= (last_g - \
                list(local_models[i].parameters())[-2].cpu().data.numpy())
        f_grads[i] = grad.reshape(-1)

    cs = smp.cosine_similarity(f_grads) - np.eye(m)
    cs = get_pca(cs)
    centroid = np.median(cs, axis = 0)  # 计算中位数
    if epoch % 25 == 1:
        with open(filename, 'a') as f:
            f.write(str(epoch) + '\n')
            for row in cs:
                line = ",".join(map(str, row))
                f.write(f"{line}\n") 
            centroid_line = ",".join(map(str, centroid))
            f.write(centroid_line + "\n")
    scores = smp.cosine_similarity([centroid], cs)[0]

    # print('-> Update score history')
    # score_history[selected_users]+= scores
    print("cosine scores:", scores)
    q1 = np.quantile(scores, 0.25)
    trust = scores - q1 
    trust = trust/trust.max()
    trust[(trust < 0)] = 0

    local_weights = [local_model.state_dict() for local_model in local_models]
    
    return average_weights(local_weights, trust)

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])

def load_model_weight(net, weight):
    index_bias = 0
    for p_index, p in enumerate(net.parameters()):
        p.data = weight[index_bias:index_bias + p.numel()].view(p.size())
        index_bias += p.numel()

def multi(client_models, device, excluded_frequency):
    vectorize_nets = [vectorize_net(cm).detach().cpu().numpy() for cm in client_models]

    cos_dis = [0.0] * len(vectorize_nets)
    length_dis = [0.0] * len(vectorize_nets)
    manhattan_dis = [0.0] * len(vectorize_nets)

    for i, g_i in enumerate(vectorize_nets):
        for j in range(len(vectorize_nets)):
            if i != j:
                g_j = vectorize_nets[j]

                cosine_distance = float(
                    (1 - np.dot(g_i, g_j) / (np.linalg.norm(g_i) * np.linalg.norm(g_j))) ** 2)   #Compute the different value of cosine distance
                manhattan_distance = float(np.linalg.norm(g_i - g_j, ord=1))    #Compute the different value of Manhattan distance
                length_distance = np.abs(float(np.linalg.norm(g_i) - np.linalg.norm(g_j)))    #Compute the different value of Euclidean distance

                cos_dis[i] += cosine_distance
                length_dis[i] += length_distance
                manhattan_dis[i] += manhattan_distance
    
    tri_distance = np.vstack([cos_dis, manhattan_dis, length_dis]).T

    cov_matrix = np.cov(tri_distance.T)
    inv_matrix = np.linalg.inv(cov_matrix)

    ma_distances = []
    for i, g_i in enumerate(vectorize_nets):
        t = tri_distance[i]
        ma_dis = np.dot(np.dot(t, inv_matrix), t.T)
        ma_distances.append(ma_dis)
    
    scores = ma_distances
    print(scores)

    p = 0.6
    p_num = p*len(scores)
    topk_ind = np.argpartition(scores, int(p_num))[:int(p_num)]   #sort

    for i in range(len(scores)):
        if i not in topk_ind:
            excluded_frequency[i] += 1
    print(excluded_frequency)

    aggregated_grad = np.mean(np.array(vectorize_nets)[topk_ind, :], axis=0).astype(np.float32)

    aggregated_model = client_models[0]  # slicing which doesn't really matter
    load_model_weight(aggregated_model, torch.from_numpy(aggregated_grad).to(device))

    return aggregated_model