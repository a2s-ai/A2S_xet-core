use std::time::Duration;

use tokio::time::Instant;

/// A latency predictor using a numerically stable, exponentially decayed linear regression:
///
/// We fit a model of the form:
///   duration_secs ≈ base_time_secs + size_bytes * inv_throughput
/// which is equivalent to:
///   duration_secs ≈ intercept + slope * size_bytes
///
/// Internally, we use a stable, online update method based on weighted means and covariances:
/// - mean_x, mean_y: weighted means of size and duration
/// - s_xx, s_xy: exponentially decayed sums of (x - mean_x)^2 and (x - mean_x)(y - mean_y)
///
/// We apply decay on each update using exp2(-elapsed / half_life).
///
/// This avoids numerical instability from large sums and is robust to shifting distributions.
pub struct LatencyPredictor {
    sum_w: f64,
    mean_x: f64,
    mean_y: f64,
    s_xx: f64,
    s_xy: f64,

    base_time_secs: f64,
    inv_throughput: f64,
    decay_half_life_secs: f64,
    last_update: Instant,
}

impl LatencyPredictor {
    pub fn new(decay_half_life: Duration) -> Self {
        Self {
            sum_w: 0.0,
            mean_x: 0.0,
            mean_y: 0.0,
            s_xx: 0.0,
            s_xy: 0.0,
            base_time_secs: 120.0, // 2 minutes, but no real weight on this.
            inv_throughput: 0.0,
            decay_half_life_secs: decay_half_life.as_secs_f64(),
            last_update: Instant::now(),
        }
    }

    /// Updates the latency model with a new observation.
    ///
    /// Applies exponential decay to prior statistics and incorporates the new sample
    /// using a numerically stable linear regression formula.
    ///
    /// - `size_bytes`: the size of the completed transmission.
    /// - `duration`: the time taken to complete the transmission.
    /// Updates the latency model with a new observation.
    ///
    /// Applies exponential decay to prior statistics and incorporates the new sample
    /// using a numerically stable linear regression formula.
    ///
    /// - `size_bytes`: the size of the completed transmission.
    /// - `duration`: the time taken to complete the transmission.
    /// - `n_concurrent`: the number of concurrent connections at the time.
    pub fn update(&mut self, size_bytes: usize, duration: Duration, avg_concurrent: f64) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        let decay = (-elapsed / self.decay_half_life_secs).exp2();

        // Feature x: number of bytes transferred in this time, assuming that multiple similar
        // connections are active.  This is just a way to treat the
        let x = (size_bytes as f64) * avg_concurrent.max(1.);

        // Target y: the time it would take to transfer x bytes, i.e. secs / byte.
        let y = duration.as_secs_f64().max(1e-6);

        // Decay previous statistics
        self.sum_w *= decay;
        self.s_xx *= decay;
        self.s_xy *= decay;

        // Update means with numerically stable method
        let weight = 1.0;
        let new_sum_w = self.sum_w + weight;
        let delta_x = x - self.mean_x;
        let delta_y = y - self.mean_y;

        let mean_x_new = self.mean_x + (weight * delta_x) / new_sum_w;
        let mean_y_new = self.mean_y + (weight * delta_y) / new_sum_w;

        self.s_xx += weight * delta_x * (x - mean_x_new);
        self.s_xy += weight * delta_x * (y - mean_y_new);

        self.mean_x = mean_x_new;
        self.mean_y = mean_y_new;
        self.sum_w = new_sum_w;

        if self.s_xx > 1e-8 {
            let slope = self.s_xy / self.s_xx;
            let intercept = self.mean_y - slope * self.mean_x;

            self.base_time_secs = intercept;
            self.inv_throughput = slope;
        } else {
            self.base_time_secs = self.mean_y;
            self.inv_throughput = 0.0;
        }

        self.last_update = now;
    }

    /// Predicts the expected completion time for a given transfer size and concurrency level.
    ///
    /// First predicts the overall latency of a transfer, assuming that there is no concurrency and
    /// connections scale with
    ///
    /// to reflect how concurrency reduces per-transfer time under stable throughput.
    ///
    /// - `size_bytes`: the size of the transfer.
    /// - `n_concurrent`: the number of concurrent connections.
    pub fn predicted_latency(&self, size_bytes: u64, avg_concurrent: f64) -> Duration {
        let predicted_secs_without_concurrency = self.base_time_secs + size_bytes as f64 * self.inv_throughput;
        let predicted_secs = predicted_secs_without_concurrency * avg_concurrent.max(1.);
        Duration::from_secs_f64(predicted_secs)
    }

    pub fn predicted_bandwidth(&self) -> f64 {
        let query_bytes = 10 * 1024 * 1024;

        // How long would it take to transmit this at full bandwidth
        let min_latency = self.predicted_latency(query_bytes, 1.);

        // Report bytes per sec in this model.
        query_bytes as f64 / min_latency.as_secs_f64().max(1e-6)
    }
}

#[cfg(test)]
mod tests {
    use tokio::time::{self, Duration as TokioDuration};

    use super::*;

    #[test]
    fn test_estimator_update() {
        let mut estimator = LatencyPredictor::new(Duration::from_secs_f64(10.0));
        estimator.update(1_000_000, Duration::from_millis(500), 1.);
        let expected = estimator.predicted_latency(1_000_000, 1.);
        assert!(expected.as_secs_f64() > 0.0);
    }

    #[test]
    fn test_converges_to_constant_observation() {
        let mut predictor = LatencyPredictor::new(Duration::from_secs_f64(10.0));
        for _ in 0..10 {
            predictor.update(1000, Duration::from_secs_f64(1.0), 1.);
        }
        let prediction = predictor.predicted_latency(1000, 1.);
        assert!((prediction.as_secs_f64() - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_decay_weighting_effect() {
        time::pause();
        let mut predictor = LatencyPredictor::new(Duration::from_secs_f64(2.0));
        predictor.update(1000, Duration::from_secs_f64(2.0), 1.);
        time::advance(TokioDuration::from_secs(2)).await;
        predictor.update(1000, Duration::from_secs_f64(1.0), 1.);
        let predicted = predictor.predicted_latency(1000, 1.).as_secs_f64();
        assert!(predicted > 1.0 && predicted < 1.6);
    }

    #[test]
    fn test_scaling_with_concurrency() {
        let mut predictor = LatencyPredictor::new(Duration::from_secs_f64(10.0));
        for _ in 0..10 {
            predictor.update(1000, Duration::from_secs_f64(1.0), 1.);
        }
        let predicted_1 = predictor.predicted_latency(1000, 1.).as_secs_f64();
        let predicted_2 = predictor.predicted_latency(1000, 2.).as_secs_f64();
        let predicted_4 = predictor.predicted_latency(1000, 4.).as_secs_f64();
        assert!(predicted_2 > predicted_1);
        assert!(predicted_4 > predicted_2);
    }
}
