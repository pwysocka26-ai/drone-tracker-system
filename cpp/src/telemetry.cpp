#include "dtracker/telemetry.hpp"

#include <iomanip>
#include <sstream>

namespace dtracker {

TelemetryWriter::TelemetryWriter(const std::string& path) : ofs_(path) {}

TelemetryWriter::~TelemetryWriter() { close(); }

void TelemetryWriter::close() {
    if (ofs_.is_open()) ofs_.close();
}

static void write_bbox_json(std::ostream& os, const BBox& b) {
    os << "[" << b.x1 << "," << b.y1 << "," << b.x2 << "," << b.y2 << "]";
}

static void write_track_json(std::ostream& os, const Track& t) {
    os << "{";
    os << "\"track_id\":" << t.track_id;
    os << ",\"raw_id\":" << t.raw_id;
    os << ",\"bbox\":";
    write_bbox_json(os, t.bbox);
    os << ",\"conf\":" << t.confidence;
    os << ",\"hits\":" << t.hits;
    os << ",\"missed_frames\":" << t.missed_frames;
    os << ",\"cx\":" << t.center.x;
    os << ",\"cy\":" << t.center.y;
    os << ",\"vx\":" << t.velocity.x;
    os << ",\"vy\":" << t.velocity.y;
    os << ",\"is_confirmed\":" << (t.is_confirmed ? "true" : "false");
    os << "}";
}

void TelemetryWriter::write(const FrameTelemetry& r) {
    if (!ofs_.is_open()) return;
    ofs_ << std::fixed << std::setprecision(4);
    ofs_ << "{";
    ofs_ << "\"frame_idx\":" << r.frame_idx;
    ofs_ << ",\"time_s\":" << r.time_s;
    ofs_ << ",\"selected_id\":";
    if (r.selected_id) ofs_ << *r.selected_id; else ofs_ << "null";
    ofs_ << ",\"persistent_owner_id\":";
    if (r.persistent_owner_id >= 0) ofs_ << r.persistent_owner_id; else ofs_ << "null";
    ofs_ << ",\"narrow_synthetic_hold\":" << (r.narrow_synthetic_hold ? "true" : "false");
    ofs_ << ",\"narrow_hold_count\":" << r.narrow_hold_count;
    ofs_ << ",\"narrow_has_owner\":" << (r.narrow_has_owner ? "true" : "false");
    ofs_ << ",\"narrow_smooth_size\":" << r.narrow_smooth_size;
    ofs_ << ",\"narrow_crop\":[" << r.narrow_crop_x1 << "," << r.narrow_crop_y1
         << "," << r.narrow_crop_x2 << "," << r.narrow_crop_y2 << "]";
    ofs_ << ",\"narrow_rendered\":" << (r.narrow_rendered ? "true" : "false");
    ofs_ << ",\"csrt_active\":" << (r.csrt_active ? "true" : "false");
    ofs_ << ",\"csrt_updated\":" << (r.csrt_updated_this_frame ? "true" : "false");
    ofs_ << ",\"csrt_synthetic_used\":" << (r.csrt_synthetic_used ? "true" : "false");
    ofs_ << ",\"csrt_score\":" << r.csrt_score;
    ofs_ << ",\"active_track_id\":";
    if (r.active_track) ofs_ << r.active_track->track_id; else ofs_ << "null";
    ofs_ << ",\"active_track_bbox\":";
    if (r.active_track) write_bbox_json(ofs_, r.active_track->bbox); else ofs_ << "null";
    ofs_ << ",\"active_track_conf\":";
    if (r.active_track) ofs_ << r.active_track->confidence; else ofs_ << "null";
    ofs_ << ",\"narrow_lock_state\":\"" << to_string(r.lock_state) << "\"";
    ofs_ << ",\"multi_tracks\":" << r.multi_tracks.size();
    ofs_ << ",\"narrow_center\":";
    if (r.narrow_center) {
        ofs_ << "[" << r.narrow_center->x << "," << r.narrow_center->y << "]";
    } else {
        ofs_ << "null";
    }
    ofs_ << ",\"center_lock\":" << (r.center_lock ? "true" : "false");
    ofs_ << ",\"inference_ms\":" << r.inference_ms;
    ofs_ << ",\"tracker_ms\":" << r.tracker_ms;
    // full tracks array (opcjonalnie, moze dazyc JSON)
    ofs_ << ",\"tracks\":[";
    for (size_t i = 0; i < r.multi_tracks.size(); ++i) {
        if (i > 0) ofs_ << ",";
        write_track_json(ofs_, r.multi_tracks[i]);
    }
    ofs_ << "]";
    ofs_ << "}\n";
    ofs_.flush();
}

}  // namespace dtracker
