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
        .expect("bind ring endpoint")
}

/// 3 local iroh endpoints in a ring: 0->1->2->0.
/// Send a Pong message around the ring and verify delivery.
#[tokio::test]
async fn test_ring_connect_3_nodes() {
    let ep0 = ring_endpoint().await;
    let ep1 = ring_endpoint().await;
    let ep2 = ring_endpoint().await;

    let addr0 = ep0.addr();
    let addr1 = ep1.addr();
    let addr2 = ep2.addr();

    // Barrier to keep all endpoints alive until everyone has finished
    let barrier = Arc::new(Barrier::new(3));

    // Node 0: send_right=1, recv_left=2. Initiates Pong, receives it back.
    let b0 = barrier.clone();
    let h0 = tokio::spawn(async move {
        let connect = async {
            let conn = ep0.connect(addr1, DRIFT_RING_ALPN).await.expect("0->1");
            let (mut send, _) = conn.open_bi().await.expect("open bi");
            write_message(&mut send, &DriftMessage::Ping).await.expect("hs");
            send
        };
        let accept = async {
            loop {
                let incoming = ep0.accept().await.expect("accept");
                let conn = incoming.await.expect("conn");
                if conn.alpn().as_ref() == DRIFT_RING_ALPN {
                    let (_, mut recv) = conn.accept_bi().await.expect("accept bi");
                    let msg = read_message(&mut recv).await.expect("read hs");
                    assert!(matches!(msg, DriftMessage::Ping));
                    return recv;
                }
            }
        };
        let (mut send_right, mut recv_left) = tokio::join!(connect, accept);

        write_message(&mut send_right, &DriftMessage::Pong).await.expect("send pong");
        let msg = read_message(&mut recv_left).await.expect("recv pong");
        assert!(matches!(msg, DriftMessage::Pong));

        b0.wait().await;
        ep0.close().await;
    });

    // Node 1: send_right=2, recv_left=0. Forwards Pong.
    let b1 = barrier.clone();
    let h1 = tokio::spawn(async move {
        let connect = async {
            let conn = ep1.connect(addr2, DRIFT_RING_ALPN).await.expect("1->2");
            let (mut send, _) = conn.open_bi().await.expect("open bi");
            write_message(&mut send, &DriftMessage::Ping).await.expect("hs");
            send
        };
        let accept = async {
            loop {
                let incoming = ep1.accept().await.expect("accept");
                let conn = incoming.await.expect("conn");
                if conn.alpn().as_ref() == DRIFT_RING_ALPN {
                    let (_, mut recv) = conn.accept_bi().await.expect("accept bi");
                    let msg = read_message(&mut recv).await.expect("read hs");
                    assert!(matches!(msg, DriftMessage::Ping));
                    return recv;
                }
            }
        };
        let (mut send_right, mut recv_left) = tokio::join!(connect, accept);

        let msg = read_message(&mut recv_left).await.expect("recv at 1");
        assert!(matches!(msg, DriftMessage::Pong));
        write_message(&mut send_right, &msg).await.expect("fwd at 1");

        b1.wait().await;
        ep1.close().await;
    });

    // Node 2: send_right=0, recv_left=1. Forwards Pong.
    let b2 = barrier.clone();
    let h2 = tokio::spawn(async move {
        let connect = async {
            let conn = ep2.connect(addr0, DRIFT_RING_ALPN).await.expect("2->0");
            let (mut send, _) = conn.open_bi().await.expect("open bi");
            write_message(&mut send, &DriftMessage::Ping).await.expect("hs");
            send
        };
        let accept = async {
            loop {
                let incoming = ep2.accept().await.expect("accept");
                let conn = incoming.await.expect("conn");
                if conn.alpn().as_ref() == DRIFT_RING_ALPN {
                    let (_, mut recv) = conn.accept_bi().await.expect("accept bi");
                    let msg = read_message(&mut recv).await.expect("read hs");
                    assert!(matches!(msg, DriftMessage::Ping));
                    return recv;
                }
            }
        };
        let (mut send_right, mut recv_left) = tokio::join!(connect, accept);

        let msg = read_message(&mut recv_left).await.expect("recv at 2");
        assert!(matches!(msg, DriftMessage::Pong));
        write_message(&mut send_right, &msg).await.expect("fwd at 2");

        b2.wait().await;
        ep2.close().await;
    });

    h0.await.expect("node 0");
    h1.await.expect("node 1");
    h2.await.expect("node 2");
}
