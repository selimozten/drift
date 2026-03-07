use drift_proto::{
    read_message, write_message, DriftMessage, NodeInfo, ShardAssignment, TrainConfig,
    TrainProgress, DRIFT_ALPN,
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
        .expect("bind")
}

/// Full training pipeline: coordinator sends config, node streams progress back.
#[tokio::test]
async fn test_training_pipeline() {
    let node_ep = test_endpoint().await;
    let coord_ep = test_endpoint().await;

    let node_addr = node_ep.addr();
    let node_id = node_ep.id();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();

    let epochs = 3u32;
    let steps_per_epoch = 5u64;

    // Node side: accept connection, receive config, stream progress
    let node_handle = tokio::spawn(async move {
        let incoming = node_ep.accept().await.expect("accept");
        let conn = incoming.await.expect("connection");
        let (mut send, mut recv) = conn.accept_bi().await.expect("accept bi");

        // Protocol handshake
        let msg = read_message(&mut recv).await.expect("read ping");
        assert!(matches!(msg, DriftMessage::Ping));

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

        // Receive config
        let msg = read_message(&mut recv).await.expect("read config");
        let config = match msg {
            DriftMessage::TrainConfig(c) => c,
            other => panic!("expected TrainConfig, got {:?}", other),
        };

        // Receive shard
        let msg = read_message(&mut recv).await.expect("read shard");
        let shard = match msg {
            DriftMessage::ShardAssignment(s) => s,
            other => panic!("expected ShardAssignment, got {:?}", other),
        };

        // Simulate training: stream progress back
        let mut loss = 2.5f64;
        for epoch in 0..config.epochs {
            for step in 0..steps_per_epoch {
                loss *= 0.9;
                write_message(
                    &mut send,
                    &DriftMessage::TrainProgress(TrainProgress {
                        node_id: shard.node_id.clone(),
                        epoch,
                        step: epoch as u64 * steps_per_epoch + step,
                        loss,
                        throughput_samples_per_sec: 100.0,
                    }),
                )
                .await
                .expect("write progress");
            }
        }

        // Wait for coordinator to finish reading
        let _ = done_rx.await;
        node_ep.close().await;
    });

    // Coordinator side
    let conn = coord_ep
        .connect(node_addr, DRIFT_ALPN)
        .await
        .expect("connect");

    let (mut send, mut recv) = conn.open_bi().await.expect("open bi");

    // Handshake
    write_message(&mut send, &DriftMessage::Ping)
        .await
        .expect("write ping");

    let msg = read_message(&mut recv).await.expect("read node info");
    assert!(matches!(msg, DriftMessage::NodeInfo(_)));

    // Send config and shard
    write_message(
        &mut send,
        &DriftMessage::TrainConfig(TrainConfig {
            model_path: "model.pt".to_string(),
            dataset_path: "data/".to_string(),
            batch_size: 32,
            learning_rate: 0.001,
            epochs,
        }),
    )
    .await
    .expect("write config");

    write_message(
        &mut send,
        &DriftMessage::ShardAssignment(ShardAssignment {
            node_id: node_id.to_string(),
            shard_index: 0,
            shard_start: 0,
            shard_end: 10000,
        }),
    )
    .await
    .expect("write shard");

    // Collect all progress updates
    let total_expected = epochs as u64 * steps_per_epoch;
    let mut received = Vec::new();

    for _ in 0..total_expected {
        let msg = read_message(&mut recv).await.expect("read progress");
        match msg {
            DriftMessage::TrainProgress(p) => received.push(p),
            other => panic!("expected TrainProgress, got {:?}", other),
        }
    }

    // Verify progress
    assert_eq!(received.len(), total_expected as usize);

    // Loss should decrease over time
    assert!(received.last().unwrap().loss < received.first().unwrap().loss);

    // Steps should be monotonically increasing
    for i in 1..received.len() {
        assert!(received[i].step > received[i - 1].step);
    }

    // Final epoch should be epochs - 1
    assert_eq!(received.last().unwrap().epoch, epochs - 1);

    let _ = done_tx.send(());
    node_handle.await.expect("node task");
    coord_ep.close().await;
}
