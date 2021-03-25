// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License. 

#include "image_cartoonizer.h"
#include "sample_timer.h"
#include <cmath> 

namespace TNN_NS {

ImageCartoonizerOutput::~ImageCartoonizerOutput() {}

ImageCartoonizer::~ImageCartoonizer() {}

// 对image进行预处理
std::shared_ptr<Mat> ImageCartoonizer::ProcessSDKInputMat(std::shared_ptr<Mat> input_mat,
                                                                    std::string name) {
    return TNNSDKSample::ResizeToInputShape(input_mat, name);
}

// MatConvertParam definition in <path-to-tnn>/include/tnn/utils/blob_converter.h
MatConvertParam ImageCartoonizer::GetConvertParamForInput(std::string tag) {
    MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0 / (255 * 0.5), 1.0 / (255 * 0.5), 1.0 / (255 * 0.5)};
    input_cvt_param.bias = {-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5};
    return input_cvt_param;
}

MatConvertParam ImageCartoonizer::GetConvertParamForOutput() {
    MatConvertParam output_cvt_param;
    output_cvt_param.scale = {0.5, 0.5, 0.5};
    output_cvt_param.bias = {0.5, 0.5, 0.5};
    return output_cvt_param;
}

// 构造一个指向ImageCartoonizerOutput的指针
std::shared_ptr<TNNSDKOutput> ImageCartoonizer::CreateSDKOutput() {
    return std::make_shared<ImageCartoonizerOutput>();
}



Status ImageCartoonizer::ProcessSDKOutput(std::shared_ptr<TNNSDKOutput> output_) {
    Status status = TNN_OK;
    // output_.get() 返回output保存的指针
    auto output = dynamic_cast<ImageCartoonizerOutput *>(output_.get());
    RETURN_VALUE_ON_NEQ(!output, false, 
                        Status(TNNERR_PARAM_ERR, "TNNSDKOutput is invalid"));
    auto output_mat = output->GetMat();
    MatConvertParam param = GetConvertParamForOutput();
    auto inverse_normalized_mat = Tensor2Image(output_mat, param);

    output->cartoonized_image = ImageInfo(inverse_normalized_mat);
    return status;

}

std::shared_ptr<Mat> ImageCartoonizer::Tensor2Image(std::shared_ptr<Mat> image, 
                                                    const MatConvertParam& param) {
    auto cartoonized_image = std::make_shared<Mat>(image->GetDeviceType(), 
                                                    image->GetMatType(), 
                                                    image->GetDims());
    auto image_data = static_cast<float *>(image->GetData());
    auto cartoonized_image_data = static_cast<uint8_t *>(cartoonized_image->GetData());
    auto image_dims = image->GetDims();
    auto hw = image_dims[2] * image_dims[3];
    auto channel = image_dims[1];
    for(int s=0; s<hw; ++s) {
        float c0 = (image_data[s*channel + 0] * param.scale[0] + param.bias[0]) * 255;
        float c1 = (image_data[s*channel + 1] * param.scale[1] + param.bias[1]) * 255;
        float c2 = (image_data[s*channel + 2] + param.scale[2] + param.bias[2]) * 255;
        float c3 = 255;

        cartoonized_image_data[s*channel + 0] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, c0)));
        cartoonized_image_data[s*channel + 1] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, c1)));
        cartoonized_image_data[s*channel + 2] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, c2)));
        cartoonized_image_data[s*channel + 3] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, c3)));
    }
    return cartoonized_image;
}

}