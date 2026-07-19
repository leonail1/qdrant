use std::collections::BTreeMap;
use std::sync::LazyLock;

use parking_lot::Mutex;

use crate::{BufferType, Device};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BufferTelemetryStats {
    pub current_logical_bytes: u64,
    pub current_allocation_bytes: u64,
    pub peak_logical_bytes: u64,
    pub peak_allocation_bytes: u64,
    pub live_buffers: u64,
    pub peak_live_buffers: u64,
    pub allocation_count: u64,
    pub free_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferLabelTelemetry {
    pub label: String,
    pub buffer_type: BufferType,
    pub stats: BufferTelemetryStats,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferDeviceTelemetry {
    pub device_id: u64,
    pub device_name: String,
    pub queue_index: usize,
    pub total: BufferTelemetryStats,
    pub by_type: Vec<(BufferType, BufferTelemetryStats)>,
    pub by_label: Vec<BufferLabelTelemetry>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BufferTelemetrySnapshot {
    pub devices: Vec<BufferDeviceTelemetry>,
}

#[derive(Debug)]
struct DeviceTelemetryState {
    device_name: String,
    queue_index: usize,
    total: BufferTelemetryStats,
    by_type: BTreeMap<BufferType, BufferTelemetryStats>,
    by_label: BTreeMap<(BufferType, String), BufferTelemetryStats>,
}

static BUFFER_TELEMETRY: LazyLock<Mutex<BTreeMap<u64, DeviceTelemetryState>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

fn increase(stats: &mut BufferTelemetryStats, logical_bytes: u64, allocation_bytes: u64) {
    stats.current_logical_bytes = stats
        .current_logical_bytes
        .checked_add(logical_bytes)
        .expect("GPU buffer logical byte counter overflow");
    stats.current_allocation_bytes = stats
        .current_allocation_bytes
        .checked_add(allocation_bytes)
        .expect("GPU buffer allocation byte counter overflow");
    stats.live_buffers = stats
        .live_buffers
        .checked_add(1)
        .expect("GPU live buffer counter overflow");
    stats.allocation_count = stats
        .allocation_count
        .checked_add(1)
        .expect("GPU buffer allocation counter overflow");
    stats.peak_logical_bytes = stats.peak_logical_bytes.max(stats.current_logical_bytes);
    stats.peak_allocation_bytes = stats
        .peak_allocation_bytes
        .max(stats.current_allocation_bytes);
    stats.peak_live_buffers = stats.peak_live_buffers.max(stats.live_buffers);
}

fn decrease(
    stats: &mut BufferTelemetryStats,
    logical_bytes: u64,
    allocation_bytes: u64,
    label: &str,
) {
    if stats.current_logical_bytes < logical_bytes
        || stats.current_allocation_bytes < allocation_bytes
        || stats.live_buffers == 0
    {
        log::error!(
            "GPU buffer telemetry underflow for {label}: current_logical={}, logical={}, current_allocation={}, allocation={}, live={}",
            stats.current_logical_bytes,
            logical_bytes,
            stats.current_allocation_bytes,
            allocation_bytes,
            stats.live_buffers,
        );
        stats.current_logical_bytes = stats.current_logical_bytes.saturating_sub(logical_bytes);
        stats.current_allocation_bytes = stats
            .current_allocation_bytes
            .saturating_sub(allocation_bytes);
        stats.live_buffers = stats.live_buffers.saturating_sub(1);
    } else {
        stats.current_logical_bytes -= logical_bytes;
        stats.current_allocation_bytes -= allocation_bytes;
        stats.live_buffers -= 1;
    }
    stats.free_count = stats
        .free_count
        .checked_add(1)
        .expect("GPU buffer free counter overflow");
}

pub(crate) fn record_buffer_allocation(
    device: &Device,
    label: &str,
    buffer_type: BufferType,
    logical_bytes: usize,
    allocation_bytes: usize,
) {
    record_allocation(
        device.telemetry_id(),
        device.name(),
        device.telemetry_queue_index(),
        label,
        buffer_type,
        logical_bytes as u64,
        allocation_bytes as u64,
    );
}

#[allow(clippy::too_many_arguments)]
fn record_allocation(
    device_id: u64,
    device_name: &str,
    queue_index: usize,
    label: &str,
    buffer_type: BufferType,
    logical_bytes: u64,
    allocation_bytes: u64,
) {
    let mut telemetry = BUFFER_TELEMETRY.lock();
    let device = telemetry
        .entry(device_id)
        .or_insert_with(|| DeviceTelemetryState {
            device_name: device_name.to_owned(),
            queue_index,
            total: BufferTelemetryStats::default(),
            by_type: BTreeMap::new(),
            by_label: BTreeMap::new(),
        });
    if device.device_name != device_name || device.queue_index != queue_index {
        log::error!(
            "GPU buffer telemetry device identity collision: id={device_id}, old={:?}/{}, new={:?}/{}",
            device.device_name,
            device.queue_index,
            device_name,
            queue_index,
        );
    }
    increase(&mut device.total, logical_bytes, allocation_bytes);
    increase(
        device.by_type.entry(buffer_type).or_default(),
        logical_bytes,
        allocation_bytes,
    );
    increase(
        device
            .by_label
            .entry((buffer_type, label.to_owned()))
            .or_default(),
        logical_bytes,
        allocation_bytes,
    );
}

pub(crate) fn record_buffer_free(
    device_id: u64,
    label: &str,
    buffer_type: BufferType,
    logical_bytes: usize,
    allocation_bytes: usize,
) {
    let logical_bytes = logical_bytes as u64;
    let allocation_bytes = allocation_bytes as u64;
    let mut telemetry = BUFFER_TELEMETRY.lock();
    let Some(device) = telemetry.get_mut(&device_id) else {
        log::error!("GPU buffer telemetry missing device {device_id} while freeing {label}");
        return;
    };
    decrease(&mut device.total, logical_bytes, allocation_bytes, label);
    let Some(by_type) = device.by_type.get_mut(&buffer_type) else {
        log::error!("GPU buffer telemetry missing type while freeing {label}");
        return;
    };
    decrease(by_type, logical_bytes, allocation_bytes, label);
    let Some(by_label) = device.by_label.get_mut(&(buffer_type, label.to_owned())) else {
        log::error!("GPU buffer telemetry missing label while freeing {label}");
        return;
    };
    decrease(by_label, logical_bytes, allocation_bytes, label);
}

pub fn buffer_telemetry_snapshot() -> BufferTelemetrySnapshot {
    let telemetry = BUFFER_TELEMETRY.lock();
    BufferTelemetrySnapshot {
        devices: telemetry
            .iter()
            .map(|(&device_id, state)| BufferDeviceTelemetry {
                device_id,
                device_name: state.device_name.clone(),
                queue_index: state.queue_index,
                total: state.total.clone(),
                by_type: state
                    .by_type
                    .iter()
                    .map(|(&buffer_type, stats)| (buffer_type, stats.clone()))
                    .collect(),
                by_label: state
                    .by_label
                    .iter()
                    .map(|((buffer_type, label), stats)| BufferLabelTelemetry {
                        label: label.clone(),
                        buffer_type: *buffer_type,
                        stats: stats.clone(),
                    })
                    .collect(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Buffer, Instance};

    #[test]
    fn tracks_current_peak_type_and_label_bytes() {
        let device_id = u64::MAX - 17;
        record_allocation(
            device_id,
            "test-device",
            3,
            "vectors",
            BufferType::Storage,
            100,
            128,
        );
        record_allocation(
            device_id,
            "test-device",
            3,
            "vectors",
            BufferType::Storage,
            50,
            64,
        );
        record_buffer_free(device_id, "vectors", BufferType::Storage, 100, 128);

        let snapshot = buffer_telemetry_snapshot();
        let device = snapshot
            .devices
            .iter()
            .find(|device| device.device_id == device_id)
            .unwrap();
        assert_eq!(device.device_name, "test-device");
        assert_eq!(device.queue_index, 3);
        assert_eq!(device.total.current_logical_bytes, 50);
        assert_eq!(device.total.current_allocation_bytes, 64);
        assert_eq!(device.total.peak_logical_bytes, 150);
        assert_eq!(device.total.peak_allocation_bytes, 192);
        assert_eq!(device.total.live_buffers, 1);
        assert_eq!(device.total.peak_live_buffers, 2);
        assert_eq!(device.total.allocation_count, 2);
        assert_eq!(device.total.free_count, 1);
        assert_eq!(device.by_type.len(), 1);
        assert_eq!(device.by_label.len(), 1);

        record_buffer_free(device_id, "vectors", BufferType::Storage, 50, 64);
        let snapshot = buffer_telemetry_snapshot();
        let device = snapshot
            .devices
            .iter()
            .find(|device| device.device_id == device_id)
            .unwrap();
        assert_eq!(device.total.current_logical_bytes, 0);
        assert_eq!(device.total.current_allocation_bytes, 0);
        assert_eq!(device.total.live_buffers, 0);
        assert_eq!(device.total.free_count, 2);
        assert_eq!(device.total.peak_allocation_bytes, 192);
    }

    #[test]
    fn real_buffer_lifecycle_updates_ledger() {
        let instance = Instance::builder().build().unwrap();
        let device = Device::new(instance.clone(), &instance.physical_devices()[0]).unwrap();
        let device_id = device.telemetry_id();
        let label = "buffer-telemetry-real-lifecycle";

        let buffer = Buffer::new(device, label, BufferType::Storage, 4096).unwrap();
        let allocation_size = buffer.allocation_size() as u64;
        assert!(allocation_size >= 4096);

        let snapshot = buffer_telemetry_snapshot();
        let device = snapshot
            .devices
            .iter()
            .find(|device| device.device_id == device_id)
            .unwrap();
        let label_stats = &device
            .by_label
            .iter()
            .find(|entry| entry.label == label && entry.buffer_type == BufferType::Storage)
            .unwrap()
            .stats;
        assert_eq!(label_stats.current_logical_bytes, 4096);
        assert_eq!(label_stats.current_allocation_bytes, allocation_size);
        assert_eq!(label_stats.live_buffers, 1);
        assert_eq!(label_stats.allocation_count, 1);
        assert_eq!(label_stats.free_count, 0);

        drop(buffer);
        let snapshot = buffer_telemetry_snapshot();
        let device = snapshot
            .devices
            .iter()
            .find(|device| device.device_id == device_id)
            .unwrap();
        let label_stats = &device
            .by_label
            .iter()
            .find(|entry| entry.label == label && entry.buffer_type == BufferType::Storage)
            .unwrap()
            .stats;
        assert_eq!(label_stats.current_logical_bytes, 0);
        assert_eq!(label_stats.current_allocation_bytes, 0);
        assert_eq!(label_stats.live_buffers, 0);
        assert_eq!(label_stats.allocation_count, 1);
        assert_eq!(label_stats.free_count, 1);
        assert_eq!(label_stats.peak_logical_bytes, 4096);
        assert_eq!(label_stats.peak_allocation_bytes, allocation_size);
    }
}
