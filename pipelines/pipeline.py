#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import torch
from pipelines.model import AVSR
from pipelines.data.data_module import AVSRDataLoader
from pipelines.detectors.mediapipe.detector import LandmarksDetector

class InferencePipeline(torch.nn.Module):
    def __init__(self, modality, model_path, model_conf, detector="mediapipe", face_track=False, device="cpu"):
        super(InferencePipeline, self).__init__()
        self.device = torch.device(device)
        # modality configuration
        self.modality = modality
        self.dataloader = AVSRDataLoader(modality, detector=detector)
        self.model = AVSR(modality, model_path, model_conf, rnnlm=None, rnnlm_conf=None, penalty=0.0, ctc_weight=0.1, lm_weight=0.0, beam_size=40, device=device)
        if face_track and self.modality in ["video", "audiovisual"]:
            self.landmarks_detector = LandmarksDetector()
        else:
            self.landmarks_detector = None


    def process_landmarks(self, data_filename, landmarks_filename):
        if self.modality == "audio":
            return None
        if self.modality in ["video", "audiovisual"]:
            landmarks = self.landmarks_detector(data_filename)
            return landmarks


    def forward(self, data_filename, landmarks_filename=None):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        transcript = self.model.infer(data)
        return transcript

    def extract_features(self, data_filename, landmarks_filename=None, extract_resnet_feats=False):
        assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
        landmarks = self.process_landmarks(data_filename, landmarks_filename)
        data = self.dataloader.load_data(data_filename, landmarks)
        with torch.no_grad():
            if isinstance(data, tuple):
                enc_feats = self.model.model.encode(data[0].to(self.device), data[1].to(self.device), extract_resnet_feats)
            else:
                enc_feats = self.model.model.encode(data.to(self.device), extract_resnet_feats)
        return enc_feats

# class InferencePipeline(torch.nn.Module):
#     def __init__(self, config_filename, detector="retinaface", face_track=False, device="cuda:0"):
#         super(InferencePipeline, self).__init__()
#         assert os.path.isfile(config_filename), f"config_filename: {config_filename} does not exist."

#         config = ConfigParser()
#         config.read(config_filename)

#         # modality configuration
#         modality = config.get("input", "modality")

#         self.modality = modality
#         # data configuration
#         input_v_fps = config.getfloat("input", "v_fps")
#         model_v_fps = config.getfloat("model", "v_fps")

#         # model configuration
#         model_path = config.get("model","model_path")
#         model_conf = config.get("model","model_conf")

#         # language model configuration
#         rnnlm = config.get("model", "rnnlm")
#         rnnlm_conf = config.get("model", "rnnlm_conf")
#         penalty = config.getfloat("decode", "penalty")
#         ctc_weight = config.getfloat("decode", "ctc_weight")
#         lm_weight = config.getfloat("decode", "lm_weight")
#         beam_size = config.getint("decode", "beam_size")

#         self.dataloader = AVSRDataLoader(modality, speed_rate=input_v_fps/model_v_fps, detector=detector)
#         self.model = AVSR(modality, model_path, model_conf, rnnlm, rnnlm_conf, penalty, ctc_weight, lm_weight, beam_size, device)
#         if face_track and self.modality in ["video", "audiovisual"]:
#             if detector == "mediapipe":
#                 from pipelines.detectors.mediapipe.detector import LandmarksDetector
#                 self.landmarks_detector = LandmarksDetector()
#             if detector == "retinaface":
#                 from pipelines.detectors.retinaface.detector import LandmarksDetector
#                 self.landmarks_detector = LandmarksDetector(device="cuda:0")
#         else:
#             self.landmarks_detector = None


#     def process_landmarks(self, data_filename, landmarks_filename):
#         if self.modality == "audio":
#             return None
#         if self.modality in ["video", "audiovisual"]:
#             if isinstance(landmarks_filename, str):
#                 landmarks = pickle.load(open(landmarks_filename, "rb"))
#             else:
#                 landmarks = self.landmarks_detector(data_filename)
#             return landmarks


#     def forward(self, data_filename, landmarks_filename=None):
#         assert os.path.isfile(data_filename), f"data_filename: {data_filename} does not exist."
#         landmarks = self.process_landmarks(data_filename, landmarks_filename)
#         data = self.dataloader.load_data(data_filename, landmarks)
#         transcript = self.model.infer(data)
#         return transcript