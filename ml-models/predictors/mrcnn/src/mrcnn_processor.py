from model import MaskRCNN

class MrcnnRunning(MaskRCNN):

    def __init__(self, mode, config, model_dir):
        super().__init__(mode, config, model_dir)

    def preprocess_image(self):
        pass

    def identify_model(self):
        model = self.build(self.mode, self.config)
        return model
        
    def preprocess_run(self):
        pass

    def training(self):
        pass

    def predict(self):
        pass

    def postprocess_run(self):
        pass