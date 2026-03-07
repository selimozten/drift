use drift_proto::allreduce::local_allreduce;
use drift_proto::ring::{ring_allreduce, RingState};
use drift_proto::{read_message, write_message, DriftMessage, DRIFT_RING_ALPN};
use iroh::endpoint::RelayMode;
use iroh::Endpoint;
use std::sync::Arc;
use tokio::sync::Barrier;

async fn ring_endpoint() -> Endpoint {
    Endpoint::builder()
        .alpns(vec![DRIFT_RING_ALPN.to_vec()])
        .relay_mode(RelayMode::Disabled)
        .clear_address_lookup()
        .bind()
        .await
        .expect("bind")
}

/// Helper: establish a ring of N endpoints, returning (send_right, recv_left) for each.
async fn setup_ring(
    endpoints: &[Endpoint],
) -> Vec<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
    let n = endpoints.len();
    let addrs: Vec<_> = endpoints.iter().map(|ep| ep.addr()).collect();
    let barrier = Arc::new(Barrier::new(n));

    let mut handles = Vec::new();
    for i in 0..n {
        let right_addr = addrs[(i + 1) % n].clone();
        let ep = endpoints[i].clone();
        let b = barrier.clone();

        handles.push(tokio::spawn(async move {
            let connect = async {
                let conn = ep.connect(right_addr, DRIFT_RING_ALPN).await.expect("connect");
                let (mut send, _) = conn.open_bi().await.expect("open bi");
                write_message(&mut send, &DriftMessage::Ping).await.expect("hs");
                send
            };
            let accept = async {
                loop {
                    let incoming = ep.accept().await.expect("accept");
                    let conn = incoming.await.expect("conn");
                    if conn.alpn().as_ref() == DRIFT_RING_ALPN {
                        let (_, mut recv) = conn.accept_bi().await.expect("accept bi");
                        let msg = read_message(&mut recv).await.expect("read hs");
                        assert!(matches!(msg, DriftMessage::Ping));
                        return recv;
                    }
                }
            };
            let (send_right, recv_left) = tokio::join!(connect, accept);
            b.wait().await;
            (send_right, recv_left)
        }));
    }

    let mut streams = Vec::new();
    for h in handles {
        streams.push(h.await.expect("ring setup"));
    }
    streams
}

/// 3-node ring all-reduce: each node has different gradients,
/// verify all converge to the same averaged result.
#[tokio::test]
async fn test_ring_allreduce_3_nodes() {
    let ep0 = ring_endpoint().await;
    let ep1 = ring_endpoint().await;
    let ep2 = ring_endpoint().await;
    let eps = vec![ep0, ep1, ep2];

    let mut streams = setup_ring(&eps).await;
    let n = 3;

    let gradients = vec![
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0],
    ];
    let expected = local_allreduce(&gradients);

    let barrier = Arc::new(Barrier::new(n));
    let mut handles = Vec::new();

    for (i, (mut send_right, mut recv_left)) in streams.drain(..).enumerate() {
        let grad = gradients[i].clone();
        let exp = expected.clone();
        let b = barrier.clone();

        handles.push(tokio::spawn(async move {
            let state = RingState::new(i, n, grad);
            let result = ring_allreduce(state, 0, &mut send_right, &mut recv_left)
                .await
                .expect("allreduce");

            for (j, (a, e)) in result.iter().zip(exp.iter()).enumerate() {
                assert!(
                    (a - e).abs() < 1e-4,
                    "rank {} elem {}: got {} expected {}",
                    i, j, a, e,
                );
            }

            b.wait().await;
        }));
    }

    for h in handles {
        h.await.expect("allreduce task");
    }

    for ep in &eps {
        ep.close().await;
    }
}

/// 2-node ring all-reduce.
#[tokio::test]
async fn test_ring_allreduce_2_nodes() {
    let ep0 = ring_endpoint().await;
    let ep1 = ring_endpoint().await;
    let eps = vec![ep0, ep1];

    let mut streams = setup_ring(&eps).await;
    let n = 2;

    let gradients = vec![vec![2.0f32, 4.0, 6.0, 8.0], vec![10.0, 20.0, 30.0, 40.0]];
    let expected = local_allreduce(&gradients);

    let barrier = Arc::new(Barrier::new(n));
    let mut handles = Vec::new();

    for (i, (mut send_right, mut recv_left)) in streams.drain(..).enumerate() {
        let grad = gradients[i].clone();
        let exp = expected.clone();
        let b = barrier.clone();

        handles.push(tokio::spawn(async move {
            let state = RingState::new(i, n, grad);
            let result = ring_allreduce(state, 0, &mut send_right, &mut recv_left)
                .await
                .expect("allreduce");

            for (a, e) in result.iter().zip(exp.iter()) {
                assert!((a - e).abs() < 1e-4);
            }

            b.wait().await;
        }));
    }

    for h in handles {
        h.await.expect("allreduce task");
    }

    for ep in &eps {
        ep.close().await;
    }
}
