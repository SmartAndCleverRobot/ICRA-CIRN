# Zero-Shot Object Goal Visual Navigation With Class-Independent Relationship Network

This implementation is modeified based on [SSNet](https://github.com/pioneer-innovation/Zero-Shot-Object-Navigation) and [SAVN](https://github.com/allenai/savn).

The code has been implemented and tested on Ubuntu 18.04, python 3.6, PyTorch 0.6 and CUDA 10.1

## Setup

1. (Recommended) Create a virtual environment using virtualenv or conda:
```
virtualenv CIRN --python=python3.6
source CIRN/bin/activate
``` 
```
conda create -n CIRN python=3.6
conda activate CIRN
```

2. Clone the repository as:
```
git clone https://github.com/SmartAndCleverRobot/ICRA-CIRN.git
cd ICRA-CIRN
```

3. For the rest of dependencies, please run 
```
pip install -r requirements.txt --ignore-installed
```


## Data

The offline data can be found [here](https://drive.google.com/drive/folders/1i6V_t6TqaTpUdUFpOJT3y3KraJjak-sa?usp=sharing).

"data.zip" (~5 GB) contains everything needed for evalution. Please unzip it and put it into the Zero-Shot-Object-Navigation folder.

For training, please also download "train.zip" (~9 GB), and put all "Floorplan" folders into `./data/thor_v1_offline_data`

Before training, please copy files `bathroom.txt, bedroom.txt, living_room.txt, kitchen.txt` into `./data/gcn/`

## Evaluation

Note: if you are not using gpu, you can remove the argument `--gpu-ids 0`

Evaluate our model under 18/4 class split

```bash
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --load_model pretrained_models\18-4.dat \
    --model ZeroGCN  \
    --gpu-ids 2 \
    --zsd 1 \
    --split 18/4
```

Evaluate our model under 14/8 class split

```bash
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --load_model pretrained_models\14-8.dat \
    --model ZeroGCN  \
    --gpu-ids 0 \
    --zsd 1 \
    --split 14/8
```

Evaluate our model under train in kitchen and evaluate in bedroom and living-room setting
```bash
# evaluate in bedroom
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --load_model pretrained_models\log0051-train_in_kitchen_test_in_bedroom_and_living_room.dat \
    --model ZeroGCN  \
    --gpu-ids 2 \
    --zsd 1 \
    --split bedroom \
    --seen_split kitchen \
    --scene_types bedroom \
    --seen_scene_types kitchen
```

```bash
# evaluate in living-room
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --load_model pretrained_models\log0051-train_in_kitchen_test_in_bedroom_and_living_room.dat \
    --model ZeroGCN  \
    --gpu-ids 2 \
    --zsd 1 \
    --split living_room \
    --seen_split kitchen \
    --scene_types living_room \
    --seen_scene_types kitchen
```
Evaluate our model under train in living-room and evaluate in bathroom and kitchen setting

```bash
# evaluate in bathroom
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --load_model pretrained_models\log0052-train_in_living-room_test_in_bathroom_and_kitchen.dat \
    --model ZeroGCN  \
    --gpu-ids 2 \
    --zsd 1 \
    --split bathroom \
    --seen_split living_room \
    --scene_types bathroom \
    --seen_scene_types living_room
```
```bash
# evaluate in kitchen
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --load_model pretrained_models\log0052-train_in_living-room_test_in_bathroom_and_kitchen.dat \
    --model ZeroGCN  \
    --gpu-ids 1 \
    --zsd 1 \
    --split kitchen \
    --seen_split living_room \
    --scene_types kitchen \
    --seen_scene_types living_room
```

## Training

Note: the folder to save trained model should be set up before training.

Train our model under 18/4 class split

```bash
python main.py \
    --title log0050 \
    --model ZeroGCN \
    --agent_type SemanticAgent \
    --gpu-ids 0 1 2 3\
    --workers 8 \
    --vis False \
    --zsd 1 \
    --partial_reward 1 \
    --split 18/4 \
```
Train our model under 14/8 class split

```bash
python main.py \
    --title log0055 \
    --model ZeroGCN \
    --agent_type SemanticAgent \
    --gpu-ids 0 1 2 3\
    --workers 16 \
    --vis False \
    --zsd 1 \
    --partial_reward 1 \
    --split 14/8 \
```

## Cross-target and Cross-scene
Train in kitchen and evaluate in bedroom and living-room

```bash
# train in kitchen
python main.py \
    --title log0051 \
    --model ZeroGCN \
    --agent_type SemanticAgent \
    --gpu-ids 0 2 3\
    --workers 8 \
    --vis False \
    --zsd 1 \
    --partial_reward 1 \
    --split kitchen \
    --scene_types kitchen 
```

```bash
# evaluate in bedroom
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --load_model log0051 \
    --model ZeroGCN  \
    --gpu-ids 2 \
    --zsd 1 \
    --split bedroom \
    --seen_split kitchen \
    --scene_types bedroom \
    --seen_scene_types kitchen
```

```bash
# evaluate in living-room
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --load_model log0051 \
    --model ZeroGCN  \
    --gpu-ids 2 \
    --zsd 1 \
    --split bedroom \
    --seen_split kitchen \
    --scene_types bedroom \
    --seen_scene_types kitchen
```

Train in living-room and evaluate in bathroom and kitchen
```bash
# train in living-room
python main.py \
    --title log0052 \
    --model ZeroGCN \
    --agent_type SemanticAgent \
    --gpu-ids 0 1\
    --workers 8 \
    --vis False \
    --zsd 1 \
    --partial_reward 1 \
    --split living_room \
    --scene_types living_room 
```

```bash
# evaluate in bathroom
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --load_model log0052 \
    --model ZeroGCN  \
    --gpu-ids 2 \
    --zsd 1 \
    --split bathroom \
    --seen_split living_room \
    --scene_types bathroom \
    --seen_scene_types living_room
```
```bash
# evaluate in kitchen
python main.py --eval \
    --test_or_val test \
    --agent_type SemanticAgent \
    --episode_type TestValEpisode \
    --load_model log0052 \
    --model ZeroGCN  \
    --gpu-ids 1 \
    --zsd 1 \
    --split kitchen \
    --seen_split living_room \
    --scene_types kitchen \
    --seen_scene_types living_room
```
