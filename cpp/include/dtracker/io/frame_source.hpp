// IFrameSource — abstract interface dla zrodla klatek (kamera / video file / sieciowy stream).
//
// Cel: dtracker_lib operuje TYLKO na tym interfejsie -- nie wie nic o konkretnym
// hardware. Vendor / integrator dostarcza adapter (np. RtspFrameSource,
// CameraLinkFrameSource, GigEVisionFrameSource) implementujacy ten interfejs.
//
// Lifecycle:
//   1. construct
//   2. open(uri) -> bool        // RTSP URL, COM port, file path, custom URI
//   3. read(frame, timestamp)   // wolane w petli, return false na EOF lub bledzie
//   4. close()
//
// Threading: implementacje moga byc thread-safe albo nie -- dtracker pipeline
// woła read() z jednego watku, sequencyjnie. Vendor decyduje czy buforuje
// klatki w background thread (rekomendowane dla niskiej latencji RTSP).
#pragma once

#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

namespace dtracker::io {

// Metadata kamery -- read-only, dostepne po open()
struct FrameSourceInfo {
    int    width = 0;
    int    height = 0;
    double fps = 0.0;
    long   total_frames = -1;   // -1 = stream (infinite), znane dla plikow
    std::string codec;          // info, np. "H264", "MJPG"
    std::string uri;            // ostatni URI z open()
};

// Single frame + timing
struct Frame {
    cv::Mat image;          // BGR, dowolny rozmiar (info.width x info.height typowo)
    double  timestamp_s = 0.0;  // sekundy od epoki strumienia / capture
    long    frame_idx = -1;     // monotoniczny licznik klatek (od 0)
    bool    is_keyframe = false;
};

class IFrameSource {
public:
    virtual ~IFrameSource() = default;

    // Otworz zrodlo. URI moze byc:
    //   - file: "/path/to/video.mp4"
    //   - rtsp: "rtsp://user:pass@host:554/stream"
    //   - device: "/dev/video0", "0" (default camera index)
    //   - custom: "cameralink://port=1&device=PleoraCam01"
    // Vendor adapter parsuje wlasny URI scheme.
    virtual bool open(const std::string& uri) = 0;

    // Pobierz nastepna klatke. Blocking. Return false gdy EOF / bledzie / disconnect.
    // Implementacja powinna wewnetrznie NIE krasować (np. RTSP dropped frame -> 1 retry).
    virtual bool read(Frame& out) = 0;

    // Read-only metadata (po open(); UB jesli wolane przed)
    virtual const FrameSourceInfo& info() const = 0;

    // Zwolnij resources. Idempotent.
    virtual void close() = 0;

    // Czy aktualnie otwarte i sprawne. Dla streams: false po disconnect.
    virtual bool is_open() const = 0;
};

// Factory ptr -- typowy use case w aplikacji
using FrameSourcePtr = std::shared_ptr<IFrameSource>;

}  // namespace dtracker::io
