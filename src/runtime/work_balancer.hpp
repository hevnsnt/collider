/**
 * work_balancer.hpp - Per-GPU work splitter for the brain wallet dispatcher.
 *
 * v1.4.2: replace the equal-split words_per_gpu = total / num_gpus pattern
 * that left the faster GPU stalled in sync_and_collect_matches() while the
 * slower GPU finished. Maintain a per-GPU EMA of words/sec throughput and
 * weight the next batch's per-GPU slice by it. Faster GPU gets a larger
 * slice so both finish near the same wall-clock time and the sync barrier
 * stops being a downtime bottleneck on mixed-GPU rigs.
 *
 * Header-only (small, no link-time deps). Pure C++, no GPU includes; safe
 * to unit-test in isolation with synthetic throughput samples.
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

namespace collider::runtime {

class WorkBalancer {
public:
    /// One GPU's slice of the current batch's word list.
    struct Slice {
        size_t start = 0;  ///< Offset into current_words for this GPU.
        size_t count = 0;  ///< Number of words this GPU should process.
    };

    /// num_gpus: total GPU slots the dispatcher iterates.
    /// alpha:    EMA blend factor for the throughput estimate.
    ///           0.25 gives an effective ~4-batch window, responsive
    ///           enough to track a clock/PCIe state change but stable
    ///           enough that one anomalous batch does not flip the
    ///           split. Clamped to [0.0, 1.0].
    explicit WorkBalancer(size_t num_gpus, double alpha = 0.25)
        : num_gpus_(num_gpus),
          alpha_(std::clamp(alpha, 0.0, 1.0)),
          ema_words_per_sec_(num_gpus, 0.0) {}

    /// Compute the per-GPU slice for this batch.
    ///
    /// gpu_active[i] == false means skip that GPU; its slice will be
    /// {start = 0, count = 0}. The slice for the LAST Active GPU absorbs
    /// any rounding remainder so the sum of all counts equals total_words
    /// exactly.
    ///
    /// Behavior:
    ///  - First batch (no EMA history yet for any Active GPU): equal
    ///    split among Active GPUs.
    ///  - Pathological case (all Active GPUs have zero EMA, e.g. all
    ///    just toggled on): equal split among Active GPUs.
    ///  - One Active GPU returning zero EMA but others have history:
    ///    that GPU is treated as if its throughput equals the minimum
    ///    of the rest so it gets a non-zero exploratory slice and can
    ///    rebuild its estimate, rather than being starved out
    ///    permanently by a single bad sample.
    ///  - total_words < num_active_gpus (degenerate small batch):
    ///    fall back to equal split; trailing GPUs may receive 0.
    ///
    /// The returned vector has exactly num_gpus_ entries, indexed by
    /// gpu_idx, so the caller's dispatch loop can index directly.
    std::vector<Slice> split(size_t total_words,
                             const std::vector<bool>& gpu_active) {
        std::vector<Slice> out(num_gpus_);
        if (total_words == 0 || num_gpus_ == 0) {
            return out;
        }

        // Collect indices of Active GPUs (the only ones eligible for work).
        std::vector<size_t> active_idx;
        active_idx.reserve(num_gpus_);
        for (size_t i = 0; i < num_gpus_; ++i) {
            const bool active = (i < gpu_active.size()) ? gpu_active[i] : true;
            if (active) active_idx.push_back(i);
        }
        if (active_idx.empty()) {
            return out;  // nothing dispatched this batch
        }

        const size_t na = active_idx.size();

        // Degenerate: not enough words to give every Active GPU one.
        // Fall back to equal split; trailing GPUs may get 0. The caller's
        // per-GPU zero-skip already handles this safely.
        if (total_words < na) {
            return equal_split(total_words, active_idx, out);
        }

        // Find the minimum positive EMA across Active GPUs. Used as the
        // exploration fallback for any Active GPU whose EMA is 0 (i.e.,
        // never measured OR last measurement was 0). Without this, a GPU
        // that returned zero throughput once would be permanently starved
        // and could never rebuild its estimate.
        double min_positive_ema = 0.0;
        bool any_positive_ema = false;
        for (size_t idx : active_idx) {
            const double e = ema_words_per_sec_[idx];
            if (e > 0.0) {
                if (!any_positive_ema || e < min_positive_ema) {
                    min_positive_ema = e;
                }
                any_positive_ema = true;
            }
        }

        // No Active GPU has any throughput history yet (first batch ever,
        // or every Active GPU was just toggled on). Equal split.
        if (!any_positive_ema) {
            return equal_split(total_words, active_idx, out);
        }

        // Build the weight vector. A GPU with no history is treated as if
        // it had the minimum positive EMA so it gets a small probe slice;
        // its measured throughput then feeds the next split.
        std::vector<double> weights(na, 0.0);
        double sum_weights = 0.0;
        for (size_t k = 0; k < na; ++k) {
            double e = ema_words_per_sec_[active_idx[k]];
            if (e <= 0.0) e = min_positive_ema;
            weights[k] = e;
            sum_weights += e;
        }

        // Compute the per-GPU integer slice count by proportional
        // allocation. Enforce a per-GPU minimum so a tiny-weight GPU still
        // gets enough work to refresh its EMA. The minimum is set to
        // total_words / (num_active * 4); on the operator's dual-GPU rig
        // (3060 ~0.67 weight, 2060 SUPER ~0.33 weight, total_words=batch),
        // this floors the 2060 SUPER at 1/8 of the batch, well below its
        // proportional share, so it never trips for that rig and only
        // engages on more extreme weight imbalances.
        const size_t min_slice = std::max<size_t>(
            1,
            total_words / (na * 4));

        std::vector<size_t> counts(na, 0);
        size_t assigned = 0;
        for (size_t k = 0; k < na; ++k) {
            // Last Active GPU absorbs the rounding remainder so the sum
            // equals total_words exactly. This is critical: the caller's
            // chunk loop iterates per-GPU and relies on
            // sum(per_gpu_word_count) == current_words.size().
            if (k == na - 1) {
                counts[k] = (assigned < total_words) ? (total_words - assigned) : 0;
                break;
            }
            // Round to nearest rather than floor; with floor, every
            // non-last GPU is biased slightly low and the last GPU ends
            // up with a disproportionate remainder.
            double raw = (weights[k] / sum_weights) * static_cast<double>(total_words);
            size_t c = static_cast<size_t>(raw + 0.5);
            if (c < min_slice) c = min_slice;
            // Do not let one GPU's slice exceed what is left for the
            // remaining GPUs (each remaining GPU must get at least 1
            // word so the trailing-remainder branch above still produces
            // a non-negative count for the last GPU).
            const size_t remaining_gpus = na - 1 - k;
            const size_t reserved_for_others = remaining_gpus;  // at least 1 each
            const size_t max_c = (assigned + reserved_for_others < total_words)
                                     ? (total_words - assigned - reserved_for_others)
                                     : 1;
            if (c > max_c) c = max_c;
            counts[k] = c;
            assigned += c;
        }

        // Materialize the (start, count) pairs in dispatcher index order.
        // Slices are contiguous: GPU at active_idx[k] gets the kth
        // contiguous chunk of current_words.
        size_t cursor = 0;
        for (size_t k = 0; k < na; ++k) {
            const size_t gpu_idx = active_idx[k];
            out[gpu_idx].start = cursor;
            out[gpu_idx].count = counts[k];
            cursor += counts[k];
        }
        return out;
    }

    /// Update the EMA for one GPU after its dispatch completed. Call
    /// AFTER the matching sync_and_collect_matches returns (so the
    /// elapsed window covers the actual GPU work, not just the issue).
    ///
    /// words_dispatched: number of input words handed to that GPU. The
    ///   per-GPU throughput numerator is the count the rule engine
    ///   actually expanded out of those (num_passphrases), but for
    ///   split-balancing what matters is the consumed-input rate; using
    ///   raw words keeps the split tied to the same quantity that
    ///   split() distributes.
    /// elapsed: wall-clock from just-before-apply_rules_to_words_gpu to
    ///   just-after-sync_and_collect_matches. Zero-duration measurements
    ///   are ignored (caller's clock skipped or sub-tick).
    void record_throughput(size_t gpu_idx,
                           size_t words_dispatched,
                           std::chrono::nanoseconds elapsed) {
        if (gpu_idx >= num_gpus_) return;
        if (words_dispatched == 0) return;
        const double ns = static_cast<double>(elapsed.count());
        if (ns <= 0.0) return;
        const double seconds = ns * 1e-9;
        const double inst = static_cast<double>(words_dispatched) / seconds;
        if (inst <= 0.0) return;
        double& ema = ema_words_per_sec_[gpu_idx];
        if (ema <= 0.0) {
            // First measurement: seed the EMA directly. Blending against
            // 0 would under-weight the first sample by (1 - alpha) and
            // skew the second batch's split toward equal even when the
            // first batch already revealed the throughput ratio.
            ema = inst;
        } else {
            ema = alpha_ * inst + (1.0 - alpha_) * ema;
        }
    }

    /// Read-only access to the current per-GPU EMA. Diagnostic /
    /// test-only; the dispatcher does not consume this directly.
    double ema_words_per_sec(size_t gpu_idx) const noexcept {
        if (gpu_idx >= num_gpus_) return 0.0;
        return ema_words_per_sec_[gpu_idx];
    }

    size_t num_gpus() const noexcept { return num_gpus_; }

private:
    std::vector<Slice>& equal_split(size_t total_words,
                                    const std::vector<size_t>& active_idx,
                                    std::vector<Slice>& out) const {
        const size_t na = active_idx.size();
        const size_t per = total_words / na;
        size_t cursor = 0;
        for (size_t k = 0; k < na; ++k) {
            const size_t gpu_idx = active_idx[k];
            const size_t count = (k == na - 1) ? (total_words - cursor) : per;
            out[gpu_idx].start = cursor;
            out[gpu_idx].count = count;
            cursor += count;
        }
        return out;
    }

    size_t num_gpus_;
    double alpha_;
    std::vector<double> ema_words_per_sec_;  ///< 0.0 == not yet measured.
};

}  // namespace collider::runtime
