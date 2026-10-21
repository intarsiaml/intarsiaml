//! Minimal Prometheus text-format reader for the server's `/metrics`: gauges, counters with labels, and histograms
//! (bucket quantiles). Only what the TUI needs; unknown lines are skipped.

/// One sample line: `name{label="v",...} value`.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub name: String,
    pub labels: Vec<(String, String)>,
    pub value: f64,
}

impl Sample {
    pub fn label(&self, key: &str) -> Option<&str> {
        self.labels.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }
}

#[derive(Clone, Debug, Default)]
pub struct Prom {
    pub samples: Vec<Sample>,
}

/// A cumulative histogram: `(upper bound, cumulative count)` in ascending bound order, last bound `+Inf`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hist {
    pub buckets: Vec<(f64, f64)>,
    pub sum: f64,
    pub count: f64,
}

impl Prom {
    pub fn parse(text: &str) -> Prom {
        Prom { samples: text.lines().filter_map(parse_line).collect() }
    }

    /// The unlabelled sample `name`.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.samples.iter().find(|s| s.name == name && s.labels.is_empty()).map(|s| s.value)
    }

    /// All samples called `name` (any labels).
    pub fn all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Sample> + 'a {
        self.samples.iter().filter(move |s| s.name == name)
    }

    /// Histogram `name` (reads `name_bucket{le=..}`, `name_sum`, `name_count`); `None` when absent or empty of buckets.
    pub fn hist(&self, name: &str) -> Option<Hist> {
        let bucket = format!("{name}_bucket");
        let mut buckets: Vec<(f64, f64)> =
            self.all(&bucket).filter_map(|s| Some((parse_value(s.label("le")?)?, s.value))).collect();
        if buckets.is_empty() {
            return None;
        }
        buckets.sort_by(|a, b| a.0.total_cmp(&b.0));
        let count = self.get(&format!("{name}_count")).unwrap_or_else(|| buckets.last().map_or(0.0, |b| b.1));
        Some(Hist { buckets, sum: self.get(&format!("{name}_sum")).unwrap_or(0.0), count })
    }
}

impl Hist {
    /// Linear-interpolated quantile `q` in 0..=1 (Prometheus `histogram_quantile`); `None` when there are no observations.
    /// A quantile in the `+Inf` bucket is reported as the highest finite bound.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        let total = self.buckets.last().map_or(0.0, |b| b.1);
        if total <= 0.0 {
            return None;
        }
        let target = q.clamp(0.0, 1.0) * total;
        let (mut lo, mut prev) = (0.0, 0.0);
        for &(le, cum) in &self.buckets {
            if cum >= target {
                if le.is_infinite() {
                    return Some(lo);
                }
                let span = cum - prev;
                return Some(if span <= 0.0 { le } else { lo + (le - lo) * (target - prev) / span });
            }
            if le.is_finite() {
                lo = le;
            }
            prev = cum;
        }
        Some(lo)
    }

    pub fn mean(&self) -> Option<f64> {
        (self.count > 0.0).then(|| self.sum / self.count)
    }

    /// Observations made since `old` (bucket-wise difference). A counter reset or a changed bucket layout yields `self`.
    pub fn since(&self, old: &Hist) -> Hist {
        let same = self.buckets.len() == old.buckets.len()
            && self.buckets.iter().zip(&old.buckets).all(|(a, b)| a.0 == b.0 && a.1 >= b.1);
        if !same || self.count < old.count {
            return self.clone();
        }
        Hist {
            buckets: self.buckets.iter().zip(&old.buckets).map(|(a, b)| (a.0, a.1 - b.1)).collect(),
            sum: self.sum - old.sum,
            count: self.count - old.count,
        }
    }
}

fn parse_value(s: &str) -> Option<f64> {
    match s.trim() {
        "+Inf" | "Inf" => Some(f64::INFINITY),
        "-Inf" => Some(f64::NEG_INFINITY),
        t => t.parse().ok(),
    }
}

fn parse_line(line: &str) -> Option<Sample> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let (name, labels, rest) = match line.find('{') {
        Some(open) => {
            let (labels, end) = parse_labels(&line[open + 1..])?;
            (&line[..open], labels, &line[open + 1 + end..])
        }
        None => {
            let sp = line.find(char::is_whitespace)?;
            (&line[..sp], Vec::new(), &line[sp..])
        }
    };
    // value, optionally followed by a timestamp
    let value = parse_value(rest.split_whitespace().next()?)?;
    Some(Sample { name: name.to_string(), labels, value })
}

/// Parse `k="v",k2="v2"}` -> (labels, index just past the closing brace).
fn parse_labels(s: &str) -> Option<(Vec<(String, String)>, usize)> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    loop {
        while i < b.len() && (b[i] == b',' || b[i] == b' ') {
            i += 1;
        }
        if i >= b.len() {
            return None;
        }
        if b[i] == b'}' {
            return Some((out, i + 1));
        }
        let ks = i;
        while i < b.len() && b[i] != b'=' {
            i += 1;
        }
        let key = s.get(ks..i)?.trim().to_string();
        i += 1; // '='
        if b.get(i) != Some(&b'"') {
            return None;
        }
        i += 1;
        let mut val = String::new();
        loop {
            match *b.get(i)? {
                b'"' => {
                    i += 1;
                    break;
                }
                b'\\' => {
                    match *b.get(i + 1)? {
                        b'n' => val.push('\n'),
                        c => val.push(c as char),
                    }
                    i += 2;
                }
                _ => {
                    // copy one UTF-8 char
                    let ch = s.get(i..)?.chars().next()?;
                    val.push(ch);
                    i += ch.len_utf8();
                }
            }
        }
        out.push((key, val));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = r#"# HELP intarsia_requests_running Requests.
# TYPE intarsia_requests_running gauge
intarsia_requests_running 2
intarsia_requests_finished_total{reason="stop"} 5
intarsia_requests_finished_total{reason="length"} 1
intarsia_info{model="Qwen/Qwen3-8B",device="Apple \"M5\""} 1
# TYPE intarsia_ttft_seconds histogram
intarsia_ttft_seconds_bucket{le="0.1"} 0
intarsia_ttft_seconds_bucket{le="0.25"} 2
intarsia_ttft_seconds_bucket{le="0.5"} 6
intarsia_ttft_seconds_bucket{le="1"} 8
intarsia_ttft_seconds_bucket{le="+Inf"} 10
intarsia_ttft_seconds_sum 4.5
intarsia_ttft_seconds_count 10
"#;

    #[test]
    fn parses_gauges_labels_and_escapes() {
        let p = Prom::parse(TEXT);
        assert_eq!(p.get("intarsia_requests_running"), Some(2.0));
        let fin: Vec<_> = p.all("intarsia_requests_finished_total").map(|s| (s.label("reason").unwrap(), s.value)).collect();
        assert_eq!(fin, vec![("stop", 5.0), ("length", 1.0)]);
        let info = p.all("intarsia_info").next().unwrap();
        assert_eq!(info.label("device"), Some("Apple \"M5\""));
        assert_eq!(p.get("missing"), None);
    }

    #[test]
    fn quantiles_interpolate_and_clamp_at_inf() {
        let h = Prom::parse(TEXT).hist("intarsia_ttft_seconds").unwrap();
        assert_eq!(h.count, 10.0);
        // p50 -> target 5, in the (0.25, 0.5] bucket holding 4 of the 6 -> 0.25 + 0.25 * (5-2)/4
        assert!((h.quantile(0.5).unwrap() - 0.4375).abs() < 1e-9);
        // p95 lands in +Inf: report the highest finite bound
        assert_eq!(h.quantile(0.95), Some(1.0));
        assert!((h.mean().unwrap() - 0.45).abs() < 1e-9);
        assert!(Prom::parse("").hist("x").is_none());
    }

    #[test]
    fn empty_histogram_has_no_quantile() {
        let h = Hist { buckets: vec![(0.1, 0.0), (f64::INFINITY, 0.0)], sum: 0.0, count: 0.0 };
        assert_eq!(h.quantile(0.5), None);
        assert_eq!(h.mean(), None);
    }

    #[test]
    fn since_subtracts_and_survives_resets() {
        let new = Prom::parse(TEXT).hist("intarsia_ttft_seconds").unwrap();
        let mut old = new.clone();
        for b in &mut old.buckets {
            b.1 = (b.1 - 4.0).max(0.0);
        }
        old.count = 6.0;
        old.sum = 2.0;
        let d = new.since(&old);
        assert_eq!(d.count, 4.0);
        assert_eq!(d.buckets.last().unwrap().1, 4.0);
        // reset (old is bigger) -> cumulative
        assert_eq!(old.since(&new), old);
    }

    #[test]
    fn tolerates_garbage_lines() {
        let p = Prom::parse("garbage\nfoo{bar 1\nok 3\nbad{a=\"x\"} notanumber\n");
        assert_eq!(p.samples.len(), 1);
        assert_eq!(p.get("ok"), Some(3.0));
    }
}
