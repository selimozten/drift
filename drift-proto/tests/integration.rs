use drift_proto::{
    read_message, write_message, DriftMessage, NodeInfo, ShardAssignment, TrainConfig, DRIFT_ALPN,
};
use iroh::endpoint::RelayMode;
use iroh::Endpoint;

async fn test_endpoint() -> Endpoint {
    Endpoint::builder()
        .alpns(vec![DRIFT_ALPN.to_vec()])
        .relay_mode(RelayMode::Disabled)
        .clear_address_lookup()
        .bind()
        .await
        .expect("bind test endpoint")
}

/// Full handshake: coordinator connects, sends Ping, receives NodeInfo,
/// sends TrainConfig + ShardAssignment, receives Pong acknowledgment.
#[tokio::test]
async fn test_full_handshake() {
    let node_ep = test_endpoint().await;
    let coord_ep = test_endpoint().await;

    let node_addr = node_ep.addr();
    let node_id = node_ep.id();

    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();

    // Node side
    let node_handle = tokio::spawn(async move {
        let incoming = node_ep.accept().await.expect("accept");
        let conn = incoming.await.expect("connection");
        let (mut send, mut recv) = conn.accept_bi().await.expect("accept bi");

        // Read initial Ping
        let msg = read_message(&mut recv).await.expect("read ping");
        assert!(matches!(msg, DriftMessage::Ping));

        // Send NodeInfo
        write_message(
            &mut send,
            &DriftMessage::NodeInfo(NodeInfo {
                node_id: node_id.to_string(),
                gpu_name: "Test GPU".to_string(),
                gpu_vram_mb: 8192,
                gpu_compute_capability: "8.6".to_string(),
                available: true,
            }),
        )
        .await
        .expect("write node info");

        // Read TrainConfig
        let msg = read_message(&mut recv).await.expect("read config");
        assert!(matches!(msg, DriftMessage::TrainConfig(_)));

        // Read ShardAssignment
        let msg = read_message(&mut recv).await.expect("read shard");
        match msg {
            DriftMessage::ShardAssignment(s) => {
                assert_eq!(s.shard_index, 0);
                assert_eq!(s.shard_start, 0);
                assert_eq!(s.shard_end, 1000);
            }
            other => panic!("expected ShardAssignment, got {:?}", other),
        }

        // Send Pong acknowledgment
        write_message(&mut send, &DriftMessage::Pong)
            .await
            .expect("write pong");

        let _ = done_rx.await;
        node_ep.close().await;
    });

    // Coordinator side
    let conn = coord_ep
        .connect(node_addr, DRIFT_ALPN)
        .await
        .expect("connect");

    let (mut send, mut recv) = conn.open_bi().await.expect("open bi");

    // Send Ping to initiate protocol
    write_message(&mut send, &DriftMessage::Ping)
        .await
        .expect("write ping");

    // Read NodeInfo
    let msg = read_message(&mut recv).await.expect("read node info");
    let info = match msg {
        DriftMessage::NodeInfo(i) => i,
        other => panic!("expected NodeInfo, got {:?}", other),
    };
    assert_eq!(info.gpu_name, "Test GPU");
    assert_eq!(info.gpu_vram_mb, 8192);

    // Send TrainConfig
    write_message(
        &mut send,
        &DriftMessage::TrainConfig(TrainConfig {
            model_path: "model.pt".to_string(),
            dataset_path: "data/".to_string(),
            batch_size: 64,
            learning_rate: 0.001,
            epochs: 5,
        }),
    )
    .await
    .expect("write config");

    // Send ShardAssignment
    write_message(
        &mut send,
        &DriftMessage::ShardAssignment(ShardAssignment {
            node_id: info.node_id.clone(),
            shard_index: 0,
            shard_start: 0,
            shard_end: 1000,
        }),
    )
    .await
    .expect("write shard");

    // Read acknowledgment
    let msg = read_message(&mut recv).await.expect("read pong");
    assert!(matches!(msg, DriftMessage::Pong));

    let _ = done_tx.send(());
    node_handle.await.expect("node task");
    coord_ep.close().await;
}

/// Test message serialization round-trip.
#[test]
fn test_message_encoding() {
    let msg = DriftMessage::TrainProgress(drift_proto::TrainProgress {
        node_id: "abc123".to_string(),
        epoch: 1,
        step: 100,
        loss: 0.5,
        throughput_samples_per_sec: 1234.5,
    });

    let bytes = drift_proto::encode_message(&msg).expect("encode");
    assert!(bytes.len() > 4);

    let len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    assert_eq!(len, bytes.len() - 4);

    let json: DriftMessage = serde_json::from_slice(&bytes[4..]).expect("decode json");
    match json {
        DriftMessage::TrainProgress(p) => {
            assert_eq!(p.step, 100);
            assert!((p.loss - 0.5).abs() < f64::EPSILON);
        }
        other => panic!("expected TrainProgress, got {:?}", other),
    }
}
