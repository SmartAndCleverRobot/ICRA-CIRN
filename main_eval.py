from __future__ import print_function, division

import os

os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.multiprocessing as mp

import time
import numpy as np
import random
import json
from tqdm import tqdm

from utils.net_util import ScalarMeanTracker
from runners import nonadaptivea3c_val, savn_val,nonadaptivea3c_val_unseen\
    ,nonadaptivea3c_val_seen,savn_val_unseen,savn_val_seen
from utils.misc_util import get_json_data, write_json_data

def main_eval(args, create_shared_model, init_agent):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = args.load_model

    processes = []

    res_queue = mp.Queue()
    if args.model == "SAVN":
        if args.zsd:
            print("use zsd setting !")
            args.learned_loss = True
            args.num_steps = 6
            target = savn_val_unseen
        else:
            args.learned_loss = True
            args.num_steps = 6
            target = savn_val
    else:
        if args.zsd:
            print("use zsd setting !")
            args.learned_loss = False
            args.num_steps = 50
            target = nonadaptivea3c_val_unseen
        else:
            args.learned_loss = False
            args.num_steps = 50
            target = nonadaptivea3c_val

    rank = 0
    for scene_type in args.scene_types:
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                model_to_open,
                create_shared_model,
                init_agent,
                res_queue,
                250,
                scene_type,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0
    train_scalars = ScalarMeanTracker()

    train_scalars_ba = ScalarMeanTracker()
    train_scalars_be = ScalarMeanTracker()
    train_scalars_k = ScalarMeanTracker()
    train_scalars_l = ScalarMeanTracker()

    proc = len(args.scene_types)
    pbar = tqdm(total=250 * proc)

    try:
        while end_count < proc:
            train_result = res_queue.get()
            pbar.update(1)
            count += 1
            if (args.scene_types[end_count] == 'bathroom'):
                train_scalars_ba.add_scalars(train_result)
            if (args.scene_types[end_count] == 'bedroom'):
                train_scalars_be.add_scalars(train_result)
            if (args.scene_types[end_count] == 'kitchen'):
                train_scalars_k.add_scalars(train_result)
            if (args.scene_types[end_count] == 'living_room'):
                train_scalars_l.add_scalars(train_result)
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)


        tracked_means = train_scalars.pop_and_reset()

        tracked_means_ba = train_scalars_ba.pop_and_reset()
        tracked_means_be = train_scalars_be.pop_and_reset()
        tracked_means_k = train_scalars_k.pop_and_reset()
        tracked_means_l = train_scalars_l.pop_and_reset()

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()

    with open(args.results_json, "w") as fp:
        json.dump(tracked_means, fp, sort_keys=True, indent=4)
        print('\n')
        for keys, values in tracked_means.items():
            print(keys + ':' + str(values))

    # with open('all_data_'+args.results_json, "a+") as f:
    #     json.dump(args.load_model, f)
    #     json.dump(tracked_means, f, sort_keys=True, indent=4)
    
    if(args.room_results):
        with open('all_data_ba_'+args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_ba, f, sort_keys=True, indent=4)
    if(args.room_results):
        with open('all_data_be_'+args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_be, f, sort_keys=True, indent=4)
    if(args.room_results):
        with open('all_data_k_'+args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_k, f, sort_keys=True, indent=4)
    if(args.room_results):
        with open('all_data_l_'+args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_l, f, sort_keys=True, indent=4)


def main_eval_seen(args, create_shared_model, init_agent, load_model):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = load_model

    processes = []

    res_queue = mp.Queue()
    if args.model == "SAVN":
        if args.zsd:
            print("use zsd setting !")
            args.learned_loss = True
            args.num_steps = 6
            target = savn_val_seen
        else:
            args.learned_loss = True
            args.num_steps = 6
            target = savn_val
    else:
        if args.zsd:
            print("use zsd setting !")
            args.learned_loss = False
            args.num_steps = 50
            target = nonadaptivea3c_val_seen
        else:
            args.learned_loss = False
            args.num_steps = 50
            target = nonadaptivea3c_val

    rank = 0
    for scene_type in args.scene_types:
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                model_to_open,
                create_shared_model,
                init_agent,
                res_queue,
                250,
                scene_type,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0
    train_scalars = ScalarMeanTracker()

    train_scalars_ba = ScalarMeanTracker()
    train_scalars_be = ScalarMeanTracker()
    train_scalars_k = ScalarMeanTracker()
    train_scalars_l = ScalarMeanTracker()

    proc = len(args.scene_types)
    pbar = tqdm(total=250 * proc)

    try:
        while end_count < proc:
            train_result = res_queue.get()
            pbar.update(1)
            count += 1
            if (args.scene_types[end_count] == 'bathroom'):
                train_scalars_ba.add_scalars(train_result)
            if (args.scene_types[end_count] == 'bedroom'):
                train_scalars_be.add_scalars(train_result)
            if (args.scene_types[end_count] == 'kitchen'):
                train_scalars_k.add_scalars(train_result)
            if (args.scene_types[end_count] == 'living_room'):
                train_scalars_l.add_scalars(train_result)
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)

        tracked_means = train_scalars.pop_and_reset()

        tracked_means_ba = train_scalars_ba.pop_and_reset()
        tracked_means_be = train_scalars_be.pop_and_reset()
        tracked_means_k = train_scalars_k.pop_and_reset()
        tracked_means_l = train_scalars_l.pop_and_reset()

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()
    model = 'seen' + load_model.split('_')[-4]
    result_file = os.path.join('result',args.load_model+'.json')
    file_data = get_json_data(result_file)
    file_data[model] = tracked_means
    write_json_data(file_data, result_file)

    # with open(args.results_json, "w") as fp:
    #     json.dump(tracked_means, fp, sort_keys=True, indent=4)
    #     print('\n')
    #     for keys, values in tracked_means.items():
    #         print(keys + ':' + str(values))

    # with open('all_data_'+args.results_json, "a+") as f:
    #     json.dump(args.load_model, f)
    #     json.dump(tracked_means, f, sort_keys=True, indent=4)

    if (args.room_results):
        with open('all_data_ba_' + args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_ba, f, sort_keys=True, indent=4)
    if (args.room_results):
        with open('all_data_be_' + args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_be, f, sort_keys=True, indent=4)
    if (args.room_results):
        with open('all_data_k_' + args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_k, f, sort_keys=True, indent=4)
    if (args.room_results):
        with open('all_data_l_' + args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_l, f, sort_keys=True, indent=4)
    
    return 

def main_eval_unseen(args, create_shared_model, init_agent, load_model):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    model_to_open = load_model

    processes = []

    res_queue = mp.Queue()
    if args.model == "SAVN":
        if args.zsd:
            print("use zsd setting !")
            args.learned_loss = True
            args.num_steps = 6
            target = savn_val_unseen
        else:
            args.learned_loss = True
            args.num_steps = 6
            target = savn_val
    else:
        if args.zsd:
            print("use zsd setting !")
            args.learned_loss = False
            args.num_steps = 50
            target = nonadaptivea3c_val_unseen
        else:
            args.learned_loss = False
            args.num_steps = 50
            target = nonadaptivea3c_val

    rank = 0
    # nonadaptivea3c_val_unseen(0, args, model_to_open, create_shared_model, init_agent, res_queue, 250,'bedroom')
    for scene_type in args.scene_types:
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                model_to_open,
                create_shared_model,
                init_agent,
                res_queue,
                250,
                scene_type,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)
        rank += 1

    count = 0
    end_count = 0
    train_scalars = ScalarMeanTracker()

    train_scalars_ba = ScalarMeanTracker()
    train_scalars_be = ScalarMeanTracker()
    train_scalars_k = ScalarMeanTracker()
    train_scalars_l = ScalarMeanTracker()

    proc = len(args.scene_types)
    pbar = tqdm(total=250 * proc)

    try:
        while end_count < proc:
            train_result = res_queue.get()
            pbar.update(1)
            count += 1
            if (args.scene_types[end_count] == 'bathroom'):
                train_scalars_ba.add_scalars(train_result)
            if (args.scene_types[end_count] == 'bedroom'):
                train_scalars_be.add_scalars(train_result)
            if (args.scene_types[end_count] == 'kitchen'):
                train_scalars_k.add_scalars(train_result)
            if (args.scene_types[end_count] == 'living_room'):
                train_scalars_l.add_scalars(train_result)
            if "END" in train_result:
                end_count += 1
                continue
            train_scalars.add_scalars(train_result)

        tracked_means = train_scalars.pop_and_reset()

        tracked_means_ba = train_scalars_ba.pop_and_reset()
        tracked_means_be = train_scalars_be.pop_and_reset()
        tracked_means_k = train_scalars_k.pop_and_reset()
        tracked_means_l = train_scalars_l.pop_and_reset()

    finally:
        for p in processes:
            time.sleep(0.1)
            p.join()

    model = 'unseen' + load_model.split('_')[-4]
    result_file = os.path.join('result',args.load_model+'.json')
    file_data = get_json_data(result_file)
    file_data[model] = tracked_means
    write_json_data(file_data, result_file)

    # 源代码
    # with open(args.results_json, "w") as fp:
    #     json.dump(tracked_means, fp, sort_keys=True, indent=4)
    #     print('\n')
    #     for keys, values in tracked_means.items():
    #         print(keys + ':' + str(values))

    # with open('all_data_'+args.results_json, "a+") as f:
    #     json.dump(args.load_model, f)
    #     json.dump(tracked_means, f, sort_keys=True, indent=4)

    if (args.room_results):
        with open('all_data_ba_' + args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_ba, f, sort_keys=True, indent=4)
    if (args.room_results):
        with open('all_data_be_' + args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_be, f, sort_keys=True, indent=4)
    if (args.room_results):
        with open('all_data_k_' + args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_k, f, sort_keys=True, indent=4)
    if (args.room_results):
        with open('all_data_l_' + args.results_json, "a+") as f:
            json.dump(args.load_model, f)
            json.dump(tracked_means_l, f, sort_keys=True, indent=4)


def full_main_eval_unseen(args, create_shared_model, init_agent):
    trained_model = args.save_model_dir
    fold = os.path.join(trained_model, args.load_model) 
    dat_file =[i for i in os.listdir(fold) if i.split('.')[1] == 'dat'] 
    for load_model in dat_file:
        load_model = os.path.join(trained_model, args.load_model, load_model) 
        main_eval_unseen(args, create_shared_model, init_agent, load_model)

def full_main_eval_seen(args, create_shared_model, init_agent):
    trained_model = args.save_model_dir
    fold = os.path.join(trained_model, args.load_model) 
    dat_file =[i for i in os.listdir(fold) if i.split('.')[1] == 'dat'] 
    for load_model in dat_file:
        load_model = os.path.join(trained_model, args.load_model, load_model) 
        main_eval_seen(args, create_shared_model, init_agent, load_model)