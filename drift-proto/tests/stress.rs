use drift_proto::{
    read_message, write_message, DriftMessage, GradientPayload, TrainProgress, DRIFT_ALPN,
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

/// Stress test: send 1000 progress messages and verify all arrive in order.
#[tokio::test]
async fn stress_many_progress_messages() {
    let receiver_ep = test_endpoint().await;
    let sender_ep = test_endpoint().await;

    let receiver_addr = receiver_ep.addr();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();

    let message_count = 1000u64;

    // Receiver side
    let recv_handle = tokio::spawn(async move {
        let incoming = receiver_ep.accept().await.expect("accept");
        let conn = incoming.await.expect("connection");
        let (mut send, mut recv) = conn.accept_bi().await.expect("accept bi");

        // Initial handshake
        let msg = read_message(&mut recv).await.expect("read ping");
        assert!(matches!(msg, DriftMessage::Ping));
        write_message(&mut send, &DriftMessage::Pong).await.expect("write pong");

        // Receive all progress messages
        let mut received = Vec::new();
        for _ in 0..message_count {
            let msg = read_message(&mut recv).await.expect("read progress");
            match msg {
                DriftMessage::TrainProgress(p) => received.push(p),
                other => panic!("expected TrainProgress, got {:?}", other),
            }
        }

        let _ = done_rx.await;
        receiver_ep.close().await;
        received
    });

    // Sender side
    let conn = sender_ep
        .connect(receiver_addr, DRIFT_ALPN)
        .await
        .expect("connect");
    let (mut send, mut recv) = conn.open_bi().await.expect("open bi");

    // Handshake
    write_message(&mut send, &DriftMessage::Ping).await.expect("write ping");
    let msg = read_message(&mut recv).await.expect("read pong");
    assert!(matches!(msg, DriftMessage::Pong));

    // Send 1000 progress messages
    for i in 0..message_count {
        let progress = TrainProgress {
            node_id: "stress-test-node".to_string(),
            epoch: (i / 100) as u32,
            step: i,
            loss: 2.5 * (0.99_f64).powi(i as i32),
            throughput_samples_per_sec: 500.0,
        };
        write_message(&mut send, &DriftMessage::TrainProgress(progress))
            .await
            .expect("write progress");
    }

    let _ = done_tx.send(());
    let received = recv_handle.await.expect("recv task");

    assert_eq!(received.len(), message_count as usize);

    // Verify ordering
    for (i, p) in received.iter().enumerate() {
        assert_eq!(p.step, i as u64);
    }

    // Verify loss decreases
    assert!(received.last().unwrap().loss < received.first().unwrap().loss);

    sender_ep.close().await;
}

/// Test sending gradient payloads (larger messages).
#[tokio::test]
async fn gradient_payload_roundtrip() {
    let receiver_ep = test_endpoint().await;
    let sender_ep = test_endpoint().await;

    let receiver_addr = receiver_ep.addr();
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();

    // 64KB gradient payload
    let gradient_data: Vec<u8> = (0..65536).map(|i| (i % 256) as u8).collect();
    let expected_data = gradient_data.clone();

    let recv_handle = tokio::spawn(async move {
        let incoming = receiver_ep.accept().await.expect("accept");
        let conn = incoming.await.expect("connection");
        let (_send, mut recv) = conn.accept_bi().await.expect("accept bi");

        // Trigger stream
        let msg = read_message(&mut recv).await.expect("read");
        let payload = match msg {
            DriftMessage::GradientPayload(g) => g,
            other => panic!("expected GradientPayload, got {:?}", other),
        };

        let _ = done_rx.await;
        receiver_ep.close().await;
        payload
    });

    let conn = sender_ep
        .connect(receiver_addr, DRIFT_ALPN)
        .await
        .expect("connect");
    let (mut send, _recv) = conn.open_bi().await.expect("open bi");

    write_message(
        &mut send,
        &DriftMessage::GradientPayload(GradientPayload {
            node_id: "grad-test".to_string(),
            step: 42,
            data: gradient_data,
        }),
    )
    .await
    .expect("write gradient");

    let _ = done_tx.send(());
    let payload = recv_handle.await.expect("recv task");

    assert_eq!(payload.node_id, "grad-test");
    assert_eq!(payload.step, 42);
    assert_eq!(payload.data, expected_data);

    sender_ep.close().await;
}
