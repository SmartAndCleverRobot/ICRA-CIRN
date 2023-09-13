import torch
from utils.net_util import gpuify
from models.model_io import ModelInput

from .agent import ThorAgent
from datasets.constants import UNSEEN_FULL_OBJECT_8CLASS_LIST,UNSEEN_FULL_OBJECT_4CLASS_LIST

class NavigationAgent(ThorAgent):  # 与SemanticAgent的不同是他在学习时删除了unseen的object detection的信息
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, create_model, args, rank, gpu_id):
        max_episode_length = args.max_episode_length
        hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space
        from utils.class_finder import episode_class

        episode_constructor = episode_class(args.episode_type)
        episode = episode_constructor(args, gpu_id, args.strict_done)

        super(NavigationAgent, self).__init__(
            create_model(args), args, rank, episode, max_episode_length, gpu_id
        )
        self.hidden_state_sz = hidden_state_sz

        if args.zsd:
            if args.split == "18/4":
                self.unseen_objects=UNSEEN_FULL_OBJECT_4CLASS_LIST
            elif args.split == "14/8":
                self.unseen_objects = UNSEEN_FULL_OBJECT_8CLASS_LIST
        else:
            self.unseen_objects=[]

    def eval_at_state(self, model_options):
        model_input = ModelInput()
        if self.episode.current_frame is None:
            model_input.state = self.state()
        else:
            model_input.state = self.episode.current_frame

        if self.episode.current_objs is None:
            model_input.objbb = self.objstate()
        else:
            model_input.objbb = self.episode.current_objs
        # remove unseen object if use zsd setting
        for unseen_object in self.unseen_objects:
            try:
                model_input.objbb.pop(unseen_object)
            except:
                continue
        model_input.hidden = self.hidden
        model_input.target_class_embedding = self.episode.glove_embedding
        model_input.action_probs = self.last_action_probs

        return model_input, self.model.forward(model_input, model_options)

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.Tensor(frame)
        return gpuify(state, self.gpu_id)

    def reset_hidden(self):
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.hidden = (
                    torch.zeros(1, self.hidden_state_sz).cuda(),
                    torch.zeros(1, self.hidden_state_sz).cuda(),
                )
        else:
            self.hidden = (
                torch.zeros(1, self.hidden_state_sz),
                torch.zeros(1, self.hidden_state_sz),
            )
        self.last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )

    def repackage_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.last_action_probs = self.last_action_probs.detach()

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())

    def exit(self):
        pass

    def objstate(self):
        return self.episode.objstate_for_agent()
