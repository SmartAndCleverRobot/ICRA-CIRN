class ModelOptions:
    def __init__(self, params=None):
        self.params = params


class ModelInput:
    """ Input to the model. """

    def __init__(
        self, state=None, hidden=None, target_class_embedding=None, action_probs=None, objbb = None, target_text = None, target_name = None, gpu_id = None
    ):
        self.state = state
        self.hidden = hidden
        self.target_class_embedding = target_class_embedding
        self.action_probs = action_probs
        self.objbb = objbb
        self.all_object = None
        self.target_text = target_text
        self.target_name = target_name
        self.gpu_id = gpu_id


class ModelOutput:
    """ Output from the model. """

    def __init__(self, value=None, logit=None, hidden=None, embedding=None):

        self.value = value
        self.logit = logit
        self.hidden = hidden
        self.embedding = embedding
