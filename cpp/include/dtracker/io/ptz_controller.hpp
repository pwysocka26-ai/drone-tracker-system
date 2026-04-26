// IPTZController — abstract interface dla sterowania PTZ (Pan-Tilt-Zoom).
//
// Cel: dtracker_lib steruje glowica przez ten interfejs -- nie wie nic
// o konkretnym protokole (PELCO-D / ONVIF / VISCA / custom serial / ROS).
// Vendor / integrator dostarcza adapter implementujacy ten interfejs.
//
// Jednostki:
//   - kat: stopnie (deg). Pan: 0=front, +90=right, -90=left, ±180=back.
//   - velocity: stopnie/s. + = right/up, - = left/down.
//   - zoom: krotnosc (1.0 = no zoom, 2.0 = 2x, 19.5 = max typowy PTZ).
//
// Lifecycle:
//   1. construct
//   2. connect(uri) -> bool
//   3. set_pan_tilt_velocity / absolute_pan_tilt / set_zoom -- komendy
//   4. get_state() -> aktualny pan/tilt/zoom + timestamp telemetrii
//   5. disconnect()
//
// Threading: implementacje powinny byc thread-safe -- dtracker pipeline
// moze wolac komendy z control thread, telemetry odbierac z osobnego
// background thread (typowy pattern dla async PTZ protocols).
#pragma once

#include <memory>
#include <string>

namespace dtracker::io {

// Read-only state PTZ -- zwracane przez get_state()
struct PTZState {
    double pan_deg = 0.0;
    double tilt_deg = 0.0;
    double zoom_x = 1.0;
    double timestamp_s = 0.0;   // sekundy od epoki / connect()
    bool   moving = false;      // czy aktualnie wykonuje ruch
    bool   valid = false;       // false jesli telemetry nieswieza / disconnect
};

// Capabilities -- co PTZ potrafi (negotiated po connect)
struct PTZCapabilities {
    bool   supports_velocity_control = true;     // continuous pan/tilt
    bool   supports_absolute_position = true;    // absolute pan/tilt
    bool   supports_zoom = true;
    bool   supports_telemetry = true;            // get_state() returns valid
    double pan_min_deg = -180.0;
    double pan_max_deg = +180.0;
    double tilt_min_deg = -90.0;
    double tilt_max_deg = +90.0;
    double max_pan_velocity_dps = 100.0;
    double max_tilt_velocity_dps = 60.0;
    double zoom_min_x = 1.0;
    double zoom_max_x = 19.5;
};

class IPTZController {
public:
    virtual ~IPTZController() = default;

    // Connect do PTZ. URI scheme zalezy od adaptera:
    //   - "serial:///dev/ttyUSB0?baud=2400&address=1"  (PELCO-D RS485)
    //   - "onvif://user:pass@192.168.1.100:80/onvif"
    //   - "visca://COM3?baud=9600&id=1"
    //   - "ros2://camera_gimbal_node"
    //   - "mock://"  (testowy)
    virtual bool connect(const std::string& uri) = 0;

    // Continuous velocity control. pan/tilt w stopniach/s.
    // (0, 0) = stop. Async: zwraca natychmiast, ruch trwa do nastepnej komendy.
    virtual bool set_pan_tilt_velocity(double pan_dps, double tilt_dps) = 0;

    // Absolute position. Async: zwraca natychmiast, ruch trwa do osiagniecia.
    virtual bool absolute_pan_tilt(double pan_deg, double tilt_deg) = 0;

    // Zoom. zoom_x w krotnosci. Async.
    virtual bool set_zoom(double zoom_x) = 0;

    // Aktualne polozenie (z telemetrii hardware). Snapshot z momentu wywolania.
    virtual PTZState get_state() = 0;

    virtual const PTZCapabilities& capabilities() const = 0;

    virtual void disconnect() = 0;
    virtual bool is_connected() const = 0;
};

using PTZControllerPtr = std::shared_ptr<IPTZController>;

}  // namespace dtracker::io
