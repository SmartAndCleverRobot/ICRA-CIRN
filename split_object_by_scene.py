
import random
from datasets.scene_util import get_scenes
from datasets.environment import Environment

mapping = ["kitchen", "living_room", "bedroom", "bathroom"]
scenes = [get_scenes("[{}]+{}".format(num + int(num > 0), '[1-30]')) for num in [0, 1, 2, 3]]

objects_all = None
with open("/home/Newdisk/lixinting/zero-shot-data/data/gcn/objects.txt") as f: # 至少这里
    objects_all = f.readlines()

objects_all = [i.split('\n')[0] for i in objects_all]
j = 0
env = None
for s in scenes:
    o = []
    for e in s:
    # for 
        if env is None:
            env = Environment(
                offline_data_dir="/home/Newdisk/lixinting/zero-shot-data/data/thor_v1_offline_data",
                use_offline_controller=True,
                grid_size=0.25,
                images_file_name="resnet18_featuremap.hdf5",
                local_executable_path=None,
            )
            env.start(e)
        else:
            env.reset(e)

        # Randomize the start location.
        start_state = env.randomize_agent_location()
        objects = env.all_objects()

        visible_objects = [obj.split("|")[0] for obj in objects]
        intersection = [obj for obj in visible_objects if obj in objects_all]
        for i in intersection:
            if i not in o:
                o.append(i)
    with open("/home/Newdisk/lixinting/zero-shot-data/data/gcn/"+ mapping[j]+".txt",'w') as f: # 至少这里
        for i in o:
            f.writelines(i)
            f.write('\n')
    j = j+1
