//! Control message parsing/formatting for the Python subprocess IPC protocol.
//!
//! Messages are newline-delimited text on stdin/stdout:
//!   Python→Rust (child stdout): DRIFT_READY, DRIFT_ALLREDUCE, DRIFT_PROGRESS, DRIFT_DONE
//!   Rust→Python (child stdin):  DRIFT_ALLREDUCE_DONE, DRIFT_STOP

/// Messages received from the Python subprocess (parsed from stdout lines).
#[derive(Debug, PartialEq)]
pub enum PythonMessage {
    Ready,
    Allreduce { op_id: u64, num_floats: usize },
    Progress { epoch: u32, step: u64, loss: f64, throughput: f64 },
    Done,
    Unknown(String),
}

/// Parse a line from the Python subprocess stdout.
pub fn parse_python_line(line: &str) -> PythonMessage {
    let line = line.trim();
    let parts: Vec<&str> = line.split_whitespace().collect();

    match parts.first().copied() {
        Some("DRIFT_READY") => PythonMessage::Ready,
        Some("DRIFT_DONE") => PythonMessage::Done,
        Some("DRIFT_ALLREDUCE") => {
            if parts.len() >= 3 {
                if let (Some(op_id), Some(num_floats)) =
                    (parts[1].parse().ok(), parts[2].parse().ok())
                {
                    return PythonMessage::Allreduce { op_id, num_floats };
                }
            }
            PythonMessage::Unknown(line.to_string())
        }
        Some("DRIFT_PROGRESS") => {
            if parts.len() >= 5 {
                if let (Some(epoch), Some(step), Some(loss), Some(throughput)) = (
                    parts[1].parse().ok(),
                    parts[2].parse().ok(),
                    parts[3].parse().ok(),
                    parts[4].parse().ok(),
                ) {
                    return PythonMessage::Progress { epoch, step, loss, throughput };
                }
            }
            PythonMessage::Unknown(line.to_string())
        }
        _ => PythonMessage::Unknown(line.to_string()),
    }
}

/// Format a DRIFT_ALLREDUCE_DONE response for the Python subprocess.
pub fn format_allreduce_done(op_id: u64) -> String {
    format!("DRIFT_ALLREDUCE_DONE {}", op_id)
}

/// Format a DRIFT_STOP message for the Python subprocess.
pub fn format_stop() -> String {
    "DRIFT_STOP".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ready() {
        assert_eq!(parse_python_line("DRIFT_READY"), PythonMessage::Ready);
        assert_eq!(parse_python_line("DRIFT_READY\n"), PythonMessage::Ready);
    }

    #[test]
    fn test_parse_done() {
        assert_eq!(parse_python_line("DRIFT_DONE"), PythonMessage::Done);
    }

    #[test]
    fn test_parse_allreduce() {
        assert_eq!(
            parse_python_line("DRIFT_ALLREDUCE 42 1024"),
            PythonMessage::Allreduce { op_id: 42, num_floats: 1024 }
        );
    }

    #[test]
    fn test_parse_progress() {
        assert_eq!(
            parse_python_line("DRIFT_PROGRESS 2 150 0.0312 256.5"),
            PythonMessage::Progress {
                epoch: 2,
                step: 150,
                loss: 0.0312,
                throughput: 256.5,
            }
        );
    }

    #[test]
    fn test_parse_unknown() {
        assert_eq!(
            parse_python_line("[info] some log line"),
            PythonMessage::Unknown("[info] some log line".to_string())
        );
    }

    #[test]
    fn test_parse_malformed_allreduce() {
        assert!(matches!(
            parse_python_line("DRIFT_ALLREDUCE bad"),
            PythonMessage::Unknown(_)
        ));
    }

    #[test]
    fn test_format_allreduce_done() {
        assert_eq!(format_allreduce_done(42), "DRIFT_ALLREDUCE_DONE 42");
    }

    #[test]
    fn test_format_stop() {
        assert_eq!(format_stop(), "DRIFT_STOP");
    }
}
