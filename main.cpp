#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/request.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace libcamera;

std::mutex mtx;
std::condition_variable cond_var;
bool frameReady = false;
Request *completedRequest = nullptr;

void requestCompleteHandler(Request *request)
{
    std::lock_guard<std::mutex> lock(mtx);
    completedRequest = request;
    frameReady = true;
    cond_var.notify_one();
    // 重設 request 內容
    // request->reuse();
    // 重新送出
    // camera->queueRequest(request);
}

int main()
{
    CameraManager cm;
    cm.start();
    if (cm.cameras().empty()) {
        std::cerr << "No camera found\n";
        return 1;
    }

    std::shared_ptr<Camera> camera = cm.cameras()[0];
    camera->acquire();

    auto config = camera->generateConfiguration({ StreamRole::Viewfinder });
    config->at(0).size = {640, 480};
    config->at(0).pixelFormat = formats::RGB888;
    // config->at(0).bufferCount = 2;
    camera->configure(config.get());

    Stream *stream = config->at(0).stream();
    FrameBufferAllocator allocator(camera);
    allocator.allocate(stream);
    const auto &buffers = allocator.buffers(stream);
    if (buffers.size() < 2) {
        std::cerr << "At least two buffers are needed\n";
        return 1;
    }

    std::vector<void *> mappedMemory;
    std::vector<std::unique_ptr<Request>> requests;
    for (const auto &buf : buffers) {
        const FrameBuffer::Plane &plane = buf->planes()[0];
        void *mem = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
        if (mem == MAP_FAILED) {
            perror("mmap");
            return 1;
        }
        mappedMemory.push_back(mem);

        std::unique_ptr<Request> req = camera->createRequest();
        req->addBuffer(stream, buf.get());
        requests.push_back(std::move(req));
    }

    camera->requestCompleted.connect(requestCompleteHandler);
    camera->start();

    for (auto &req : requests)
        camera->queueRequest(req.get());

    cv::CascadeClassifier face_cascade;
    face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");

    std::cout << "Press ESC to exit...\n";

    while (true)
    {
        std::unique_lock<std::mutex> lock(mtx);
        cond_var.wait(lock, [] { return frameReady; });
        frameReady = false;

        const FrameBuffer *fb = completedRequest->buffers().begin()->second;
        int index = -1;
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (buffers[i].get() == fb) {
                index = i;
                break;
            }
        }
        if (index == -1)
            continue;

        cv::Mat img(480, 640, CV_8UC3, mappedMemory[index]);
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(img, faces);

        for (const auto &face : faces)
            cv::rectangle(img, face, {0, 255, 0}, 2);

        cv::imshow("Face Detection", img);
        if (cv::waitKey(1) == 27) // ESC
            break;

        camera->queueRequest(completedRequest);
    }

    for (size_t i = 0; i < mappedMemory.size(); ++i)
        munmap(mappedMemory[i], buffers[i]->planes()[0].length);

    camera->stop();
    camera->release();
    cm.stop();
    return 0;
}
