import torch
import numpy as np
import h5py

from utils.model_util import gpuify, toFloatTensor
from models.model_io import ModelInput
import CLIP.clip.clip as clip
from torchvision import transforms
from PIL import Image
from .agent import ThorAgent


class ClipAgent(ThorAgent):
    """ A navigation agent who learns with pretrained embeddings. """

    def __init__(self, create_model, args, rank, gpu_id):
        max_episode_length = args.max_episode_length
        hidden_state_sz = args.hidden_state_sz
        self.action_space = args.action_space
        from utils.class_finder import episode_class

        episode_constructor = episode_class(args.episode_type)
        episode = episode_constructor(args, gpu_id, args.strict_done)

        super(ClipAgent, self).__init__(
            create_model(args), args, rank, episode, max_episode_length, gpu_id
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("RN50", device=self.device)
        self.unloader = transforms.ToPILImage()

        self.hidden_state_sz = hidden_state_sz

    def eval_at_state(self, model_options):
        model_input = ModelInput()

        # model inputs
        if self.episode.current_frame is None:
            image = self.state()
        else:
            image = self.episode.current_frame

        model_input.hidden = self.hidden

        # current_detection_feature = self.episode.current_detection_feature()
        # current_detection_feature = current_detection_feature[self.targets_index, :]
        # target_embedding_array = np.zeros((len(self.targets), 1))
        # target_embedding_array[self.targets.index(self.episode.target_object)] = 1

        # self.episode.detection_results.append(
        #     list(current_detection_feature[self.targets.index(self.episode.target_object), 512:]))

        # target_embedding = {'appear': current_detection_feature[:, :512],
        #                     'info': current_detection_feature[:, 512:],
        #                     'indicator': target_embedding_array}
        # target_embedding['appear'] = toFloatTensor(target_embedding['appear'], self.gpu_id)
        # target_embedding['info'] = toFloatTensor(target_embedding['info'], self.gpu_id)
        # target_embedding['indicator'] = toFloatTensor(target_embedding['indicator'], self.gpu_id)
        # model_input.target_class_embedding = target_embedding
        # model_input.glove = self.episode.glove_embedding
        model_input.action_probs = self.last_action_probs
        # model_input.target_text = self.episode.target_object
        # model_input.state  model_input.hidden  model_input.action_probs  model_input.target_text
        image_copy = image.cpu().clone()
        image_copy = image_copy.transpose(0,2)
        image_copy = self.unloader(image_copy)
        image = self.preprocess(image_copy).unsqueeze(0).to(self.device)
        text_str = "Navigate to a {}".format(self.episode.target_object)
        text = clip.tokenize(text_str).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image) 
            text_features = self.clip_model.encode_text(text) 
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)          

        model_input.state = image_features
        model_input.target_text = text_features

        return model_input, self.model.forward(model_input, model_options)

    def preprocess_frame(self, frame):
        """ Preprocess the current frame for input into the model. """
        state = torch.Tensor(frame)
        return gpuify(state, self.gpu_id)

    def reset_hidden(self):
        with torch.cuda.device(self.gpu_id):
            self.hidden = (
                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
                torch.zeros(2, 1, self.hidden_state_sz).cuda(),
            )

        self.last_action_probs = gpuify(
            torch.zeros((1, self.action_space)), self.gpu_id
        )

    def repackage_hidden(self):
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        self.last_action_probs = self.last_action_probs.detach()

    def state(self):
        return self.preprocess_frame(self.episode.state_for_agent())  # 把这里改成对真实图像的处理
        # return self.preprocess_frame(self.episode.rgb_state_for_agent())

    def exit(self):
        pass
