#include "image_cartoonizer.h"
#include "image_cartoonizer_jni.h"
#include "helper_jni.h"
#include <android/bitmap.h>

static std::shared_ptr<TNN_NS::ImageCartoonizer> gCartoonizer;
static int gComputeUnitType = 0;
static jclass clsImageInfo;
static jmethodID midconstructorImageInfo;
static jfieldID fidimage_width;
static jfieldID fidimage_height;
static jfieldID fidimage_channel;
static jfieldID fiddata;

JNIEXPORT JNICALL jint TNN_CARTOONIZE(init)(JNIEnv *env, jobject thiz, 
                                            jstring modelPath, jint width, 
                                            jint height, jint computeUnitType)
{
    setBenchResult("");
    std::vector<int> nchw = {1, 3, height, width};
    gCartoonizer = std::make_shared<TNN_NS::ImageCartoonizer>();
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/SimpleGenerator.opt.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/SimpleGenerator.opt.tnnmodel");
    LOGI("proto content size %d model content size %d", protoContent.length(), 
                                                        modelContent.length());
    TNN_NS::Status status = TNN_NS::TNN_OK;
    gComputeUnitType = computeUnitType;

    auto option = std::make_shared<TNN_NS::TNNSDKOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    option->input_shapes = {};
    option->library_path="";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    if (gComputeUnitType == 1) {
        option->compute_units = TNN_NS::TNNComputeUnitsGPU;
    } else if (gComputeUnitType == 2) {
        LOGI("the device type  %d device huawei_npu" ,gComputeUnitType);
        gCartoonizer->setNpuModelPath(modelPathStr + "/");
        gCartoonizer->setCheckNpuSwitch(false);
        option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
    } else {
	    option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    }
    status = gCartoonizer->Init(option);

    if (status != TNN_NS::TNN_OK) {
        LOGE("cartoonizer init failed $d", (int)status);
        return -1;
    }

    if (clsImageInfo == NULL) {
        clsImageInfo = static_cast<jclass>(env->NewGlobalRef(env->FindClass("com/tencent/tnn/demo/ImageInfo")));
        midconstructorImageInfo = env->GetMethodID(clsImageInfo, "<init>", "()V");
        fidimage_width = env->GetFieldID(clsImageInfo, "image_width" , "I");
        fidimage_height = env->GetFieldID(clsImageInfo, "image_height" , "I");
        fidimage_channel = env->GetFieldID(clsImageInfo, "image_channel" , "I");
        fiddata = env->GetFieldID(clsImageInfo, "data" , "[B");
    }

    return 0;
}

JNIEXPORT jboolean TNN_CARTOONIZE(checkNpu)(JNIEnv *env, jobject thiz, jstring modelPath) {
    TNN_NS::ImageCartoonizer tmpCartoonizer;
    std::string protoContent, modelContent;
    std::string modelPathStr(jstring2string(env, modelPath));
    protoContent = fdLoadFile(modelPathStr + "/SimpleGenerator.opt.tnnproto");
    modelContent = fdLoadFile(modelPathStr + "/SimpleGenerator.opt.tnnmodel");
    auto option = std::make_shared<TNN_NS::TNNSDKOption>();
    option->compute_units = TNN_NS::TNNComputeUnitsHuaweiNPU;
    option->library_path = "";
    option->proto_content = protoContent;
    option->model_content = modelContent;
    tmpCartoonizer.setNpuModelPath(modelPathStr + "/");
    tmpCartoonizer.setCheckNpuSwitch(true);
    TNN_NS::Status ret = tmpCartoonizer.Init(option);
    return ret == TNN_NS::TNN_OK; 
}



JNIEXPORT JNICALL jint TNN_CARTOONIZE(deinit)(JNIEnv *env, jobject thiz)
{
    gCartoonizer = nullptr;
    return 0;
}

JNIEXPORT JNICALL jobjectArray TNN_CARTOONIZE(cartoonizeFromImage)(JNIEnv *env, jobject thiz, 
                                                jobject imageSource, jint width, jint height)
{
    jobjectArray imageInfoArray;
    std::vector<TNN_NS::ImageInfo> imageInfoList;
    int ret = -1;
    AndroidBitmapInfo  sourceInfocolor;
    void*              sourcePixelscolor;

    if (AndroidBitmap_getInfo(env, imageSource, &sourceInfocolor) < 0) {
        return 0;
    }

    if (sourceInfocolor.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return 0;
    }

    if ( AndroidBitmap_lockPixels(env, imageSource, &sourcePixelscolor) < 0) {
        return 0;
    }

    TNN_NS::BenchOption bench_option;
    bench_option.forward_count = 20;
    gCartoonizer->SetBenchOption(bench_option);
    TNN_NS::DeviceType dt = TNN_NS::DEVICE_ARM;
    TNN_NS::DimsVector target_dims = {1, 3, height, width};
    std::shared_ptr<TNN_NS::Mat> input_mat = std::make_shared<TNN_NS::Mat>(dt, TNN_NS::N8UC4, 
                                                                target_dims, sourcePixelscolor);

    std::shared_ptr<TNN_NS::TNNSDKInput> input = std::make_shared<TNN_NS::TNNSDKInput>(input_mat);
    std::shared_ptr<TNN_NS::TNNSDKOutput> output = gCartoonizer->CreateSDKOutput();
    TNN_NS::Status status = gCartoonizer->Predict(input, output);

    gCartoonizer->ProcessSDKOutput(output);
    AndroidBitmap_unlockPixels(env, imageSource);

    if (status != TNN_NS::TNN_OK) {
        return 0;
    }

    imageInfoList.push_back(dynamic_cast<TNN_NS::ImageCartoonizerOutput *>(output.get())->cartoonized_image);

    imageInfoArray = env->NewObjectArray(imageInfoList.size(), clsImageInfo, NULL);
    jobject objImageInfo = env->NewObject(clsImageInfo, midconstructorImageInfo);
    int image_width = imageInfoList[0].image_width;
    int image_height = imageInfoList[0].image_height;
    int image_channel = imageInfoList[0].image_channel;
    int dataNum = image_channel * image_width * image_height;

    env->SetIntField(objImageInfo, fidimage_width, image_width);
    env->SetIntField(objImageInfo, fidimage_height, image_height);
    env->SetIntField(objImageInfo, fidimage_channel, image_channel);

    jbyteArray jarrayData = env->NewByteArray(dataNum);
    env->SetByteArrayRegion(jarrayData, 0, dataNum , (jbyte*)imageInfoList[0].data.get());
    env->SetObjectField(objImageInfo, fiddata, jarrayData);

    env->SetObjectArrayElement(imageInfoArray, 0, objImageInfo);
    env->DeleteLocalRef(objImageInfo);

    return imageInfoArray;
}