from __future__ import division

import time
import torch
import setproctitle
import copy
from datasets.glove import Glove
from datasets.data import get_data, name_to_num,get_unseen_data,get_seen_data

from models.model_io import ModelOptions

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    end_episode,
    reset_player,
    compute_spl,
    get_bucketed_metrics,
)


def nonadaptivea3c_val(
    rank,
    args,
    model_to_open,
    model_create_fn,
    initialize_agent,
    res_queue,
    max_count,
    scene_type,
):

    glove = Glove(args.glove_file)
    scenes, possible_targets, targets, rooms = get_data(args.scene_types, args.val_scenes) #mark3
    num = name_to_num(scene_type) #mark
    scenes = scenes[num]
    targets = targets[num]
    rooms = rooms[num]   #mark2

    if scene_type == "living_room":
        args.max_episode_length = 200
    else:
        args.max_episode_length = 100

    setproctitle.setproctitle("Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    shared_model = model_create_fn(args)

    if model_to_open != "":
        try:
            saved_state = torch.load(
                model_to_open, map_location=lambda storage, loc: storage
            )
            shared_model.load_state_dict(saved_state)
        except:
            shared_model.load_state_dict(model_to_open)

    player = initialize_agent(model_create_fn, args, rank, gpu_id=gpu_id)
    player.sync_with_shared(shared_model)
    count = 0

    model_options = ModelOptions()

    j = 0

    while count < max_count:

        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
         #mark 1
        new_episode(args, player, scenes, possible_targets, targets, rooms, glove=glove)
        player_start_state = copy.deepcopy(player.environment.controller.state)
        player_start_time = time.time()

        # Train on the new episode.
        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, model_options, False)
            # Compute the loss.
            loss = compute_loss(args, player, gpu_id, model_options)
            if not player.done:
                reset_player(player)

        for k in loss:
            loss[k] = loss[k].item()
        spl, best_path_length = compute_spl(player, player_start_state)

        bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success)

        end_episode(
            player,
            res_queue,
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
            spl=spl,
            **bucketed_spl,
        )

        count += 1
        reset_player(player)

        j = (j + 1) % len(args.scene_types)

    player.exit()
    res_queue.put({"END": True})


def nonadaptivea3c_val_unseen(
    rank,
    args,
    model_to_open,
    model_create_fn,
    initialize_agent,
    res_queue,
    max_count,
    scene_type,
):

    glove = Glove(args.glove_file)
    print("场景类型为：{}".format(args.scene_types))
    scenes, possible_targets, targets, rooms = get_unseen_data(args.scene_types, args.val_scenes,args.split) #mark3
    # num = name_to_num(scene_type) #mark
    num = name_to_num(args, scene_type)
    scenes = scenes[num]
    targets = targets[num]
    rooms = rooms[num]   #mark2

    if scene_type == "living_room":
        args.max_episode_length = 200
    else:
        args.max_episode_length = 100

    setproctitle.setproctitle("Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    shared_model = model_create_fn(args)

    if model_to_open != "":
        try:
            saved_state = torch.load(
                model_to_open, map_location=lambda storage, loc: storage
            )
            shared_model.load_state_dict(saved_state)
        except:
            shared_model.load_state_dict(model_to_open)

    player = initialize_agent(model_create_fn, args, rank, gpu_id=gpu_id)
    player.sync_with_shared(shared_model)
    count = 0

    model_options = ModelOptions()

    j = 0

    while count < max_count:

        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
         #mark 1
        new_episode(args, player, scenes, possible_targets, targets, rooms, glove=glove)
        player_start_state = copy.deepcopy(player.environment.controller.state)
        player_start_time = time.time()

        # Train on the new episode.
        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, model_options, False)
            # Compute the loss.
            loss = compute_loss(args, player, gpu_id, model_options)
            if not player.done:
                reset_player(player)

        for k in loss:
            loss[k] = loss[k].item()
        spl, best_path_length = compute_spl(player, player_start_state)

        bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success)

        end_episode(
            player,
            res_queue,
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
            spl=spl,
            **bucketed_spl,
        )

        count += 1
        reset_player(player)

        j = (j + 1) % len(args.scene_types)

    player.exit()
    res_queue.put({"END": True})

def nonadaptivea3c_val_seen(
    rank,
    args,
    model_to_open,
    model_create_fn,
    initialize_agent,
    res_queue,
    max_count,
    scene_type,
):

    glove = Glove(args.glove_file)
    scenes, possible_targets, targets, rooms = get_seen_data(args.scene_types, args.val_scenes,args.split) #mark3
    num = name_to_num(args, scene_type) #mark
    scenes = scenes[num]
    targets = targets[num]
    rooms = rooms[num]   #mark2

    if scene_type == "living_room":
        args.max_episode_length = 200
    else:
        args.max_episode_length = 100

    setproctitle.setproctitle("Agent: {}".format(rank))

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    shared_model = model_create_fn(args)

    if model_to_open != "":
        try:
            saved_state = torch.load(
                model_to_open, map_location=lambda storage, loc: storage
            )
            shared_model.load_state_dict(saved_state)
        except:
            shared_model.load_state_dict(model_to_open)

    player = initialize_agent(model_create_fn, args, rank, gpu_id=gpu_id)
    player.sync_with_shared(shared_model)
    count = 0

    model_options = ModelOptions()

    j = 0

    while count < max_count:

        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
         #mark 1
        new_episode(args, player, scenes, possible_targets, targets, rooms, glove=glove)
        player_start_state = copy.deepcopy(player.environment.controller.state)
        player_start_time = time.time()

        # Train on the new episode.
        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, model_options, False)
            # Compute the loss.
            loss = compute_loss(args, player, gpu_id, model_options)
            if not player.done:
                reset_player(player)

        for k in loss:
            loss[k] = loss[k].item()
        spl, best_path_length = compute_spl(player, player_start_state)

        bucketed_spl = get_bucketed_metrics(spl, best_path_length, player.success)

        end_episode(
            player,
            res_queue,
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
            spl=spl,
            **bucketed_spl,
        )

        count += 1
        reset_player(player)

        j = (j + 1) % len(args.scene_types)

    player.exit()
    res_queue.put({"END": True})