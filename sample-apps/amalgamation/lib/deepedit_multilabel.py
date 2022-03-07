# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Sequence, Union

from monai.inferers import SimpleInferer
from monai.networks.nets import DynUNet
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    SqueezeDimd,
    ToNumpyd,
)

from monailabel.deepedit.multilabel.transforms import (
    AddGuidanceFromPointsCustomd,
    AddGuidanceSignalCustomd,
    DiscardAddGuidanced,
    ResizeGuidanceMultipleLabelCustomd,
)
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored

deepedit_label_names = {
    "spleen": 1,
    "right kidney": 2,
    "left kidney": 3,
    "liver": 6,
    "stomach": 7,
    "aorta": 8,
    "inferior vena cava": 9,
    "background": 0,
}

deepedit_network_params = {
    "spatial_dims": 3,
    "in_channels": len(deepedit_label_names) + 1,  # All labels plus Image
    "out_channels": len(deepedit_label_names),  # All labels including background
    "kernel_size": [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ],
    "strides": [
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 1],
    ],
    "upsample_kernel_size": [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 1],
    ],
    "norm_name": "instance",
    "deep_supervision": False,
    "res_block": True,
}


class DeepEditSegmentation(InferTask):
    """
    This provides Inference Engine for pre-trained model over Multi Atlas Labeling Beyond The Cranial Vault (BTCV)
    dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        label_names=None,
        dimension=3,
        spatial_size=(128, 128, 64),
        target_spacing=(1.0, 1.0, 1.0),
        description="A DeepEdit model for volumetric (3D) segmentation over 3D Images",
    ):
        if label_names is None:
            label_names = deepedit_label_names
        if network is None:
            network = DynUNet(**deepedit_network_params)

        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=label_names,
            dimension=dimension,
            description=description,
            input_key="image",
            output_label_key="pred",
            output_json_key="result",
        )

        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.label_names = label_names

    def pre_transforms(self, data=None):
        return [
            LoadImaged(keys="image", reader="ITKReader"),
            AddChanneld(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            # This transform may not work well for MR images
            ScaleIntensityRanged(
                keys="image",
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
            DiscardAddGuidanced(keys="image", label_names=self.label_names),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
        ]

    def inferer(self, data=None) -> Callable:
        return SimpleInferer()

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]


class DeepEditAnnotation(InferTask):
    """
    This provides Inference Engine for Deepgrow over DeepEdit model.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.DEEPEDIT,
        dimension=3,
        description="A pre-trained 3D Deepedit model based on DynUnet",
        spatial_size=(128, 128, 128),
        target_spacing=(1.5, 1.5, 2.0),
        label_names=None,
    ):
        if label_names is None:
            label_names = deepedit_label_names
        if network is None:
            network = DynUNet(**deepedit_network_params)

        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=label_names,
            dimension=dimension,
            description=description,
            config={"result_extension": [".nrrd", ".nii.gz"]},
        )

        self.spatial_size = spatial_size
        self.target_spacing = target_spacing
        self.label_names = label_names

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image", reader="ITKReader"),
            AddChanneld(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            # This transform may not work well for MR images
            ScaleIntensityRanged(
                keys="image",
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            AddGuidanceFromPointsCustomd(ref_image="image", guidance="guidance", label_names=self.label_names),
            Resized(keys="image", spatial_size=self.spatial_size, mode="area"),
            ResizeGuidanceMultipleLabelCustomd(guidance="guidance", ref_image="image"),
            AddGuidanceSignalCustomd(keys="image", guidance="guidance"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
        ]

    def inferer(self, data=None) -> Callable:
        return SimpleInferer()

    def inverse_transforms(self, data=None) -> Union[None, Sequence[Callable]]:
        return []  # Self-determine from the list of pre-transforms provided

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
