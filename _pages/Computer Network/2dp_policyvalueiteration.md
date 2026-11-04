---
title: "[Network] 1. 네트워크 기초: Packet Switching, Circuit Switching"
date: "2026-11-04"
tags:
    - Computer Network
    - Packet Switching
    - Circuit Switching
---

> **참고한 자료**
> - [Computer Networking: a Top Down Approach (Pearson)(9th Edition)](https://gaia.cs.umass.edu/kurose_ross/)

---

# 1. 기본 용어

- **Host (End System)**: 컴퓨터, 스마트폰, 스마트워치, 게임 콘솔, 서버처럼 인터넷에 연결되어 애플리케이션을 실행하는 장치
- **Packet**: End system이 데이터를 잘게 나눈 조각에 헤더를 붙인 전송 단위
- **ISP (Internet Service Provider)**: End system에 인터넷 접속을 제공하는 주체. 통신사, 케이블 인터넷 사업자, 셀룰러 데이터 사업자뿐 아니라 대학이나 공항처럼 자체 네트워크로 접속을 제공하는 기관도 넓은 의미에서 ISP에 포함됨
- **Protocol**: 통신하는 개체들 사이에서 주고받는 메시지의 형식과 순서, 메시지를 보내거나 받았을 때 취할 동작을 정의한 **규칙**
- **RFC (Request for Comments)**: IETF에서 인터넷 기술, 프로토콜(HTTP, TCP/IP 등), 연구 결과 및 규격을 정리해 발행한 기술 문서
- **Access Network**: End system을 경로상의 첫 번째 라우터(Edge Router)에 물리적으로 연결하는 네트워크. 예: 가정용 광랜, 기업·학교 LAN, 셀룰러 네트워크

---

# 2. 네트워크의 기본적인 동작

## 2.1. Packet Switching

End system은 네트워크에서 다양한 메시지를 주고받는다. 메시지에는 이메일, 사진, 오디오 파일 같은 데이터뿐 아니라 Handshaking처럼 통신을 제어하기 위한 내용도 포함된다.

Source는 긴 메시지를 작은 데이터 조각으로 나누고, 각 조각에 헤더를 붙여 패킷을 만든다. 패킷은 communication link와 packet switch를 차례로 거쳐 destination에 도착한다. Packet switch의 대표적인 예로는 라우터가 있다.

패킷은 link의 전송률(transmission rate)로 전송된다. 길이가 $L$ bits인 패킷을 전송률이 $R$ bits/sec인 link로 내보내는 데 걸리는 시간은 다음과 같다.

$$
d_{\text{trans}} = \frac{L}{R}
$$

#### Store-and-Forward Transmission

Store-and-forward는 packet switch가 패킷의 모든 비트를 받아 저장한 뒤에야 다음 link로 전송을 시작하는 방식이다. 대부분의 packet switch가 이 방식을 사용한다.

따라서 패킷이 packet switch 하나를 거치면 전송 지연이 두 번 발생한다.

1. Source가 패킷 전체를 packet switch로 보내는 데 $L/R$초가 걸린다.
2. Packet switch가 패킷 전체를 받은 뒤 다음 link로 내보내는 데 다시 $L/R$초가 걸린다.

즉, 두 link를 거치는 데 총 $2L/R$초가 걸린다. 이를 일반화하면 source와 destination 사이에 $N$개의 link가 있을 때 end-to-end delay는 다음과 같다.

$$
d_{\text{end-to-end}} = N\frac{L}{R}
$$

> 이 식은 패킷 하나만 전송하고, 모든 link의 전송률이 $R$로 같으며, 전파, 처리 등의 다른 지연을 무시한다고 가정한 식이다.

하지만 실제 인터넷에서는 당연히 패킷이 1개씩만 보내지지 않고, 수많은 패킷이 오고간다.

#### Queuing Delays, Packet Loss

Packet switch에는 연결된 link마다 **output buffer(output queue)**가 있다. Output buffer는 해당 link로 내보낼 패킷을 임시로 저장하는 공간이다.

새로 도착한 패킷이 나가야 할 link가 다른 패킷을 전송 중이면, 그 패킷은 output buffer에서 차례를 기다린다. 이로 인해 패킷은 store-and-forward로 인한 전송 지연에 더해 queuing delay를 겪는다. Queuing delay는 buffer에 쌓인 패킷의 양에 따라 달라지므로, 네트워크 혼잡도에 따라 패킷마다 다르게 나타난다.

이러한 현상은 특정 link로 향하는 패킷의 도착 속도가 일시적으로 그 link의 전송률 $R$보다 빠를 때 두드러진다. 패킷이 나가는 속도보다 들어오는 속도가 빠르므로 buffer에 패킷이 쌓이기 때문이다.

Buffer 공간은 유한하므로, 패킷이 도착했을 때 output buffer가 가득 차 있으면 패킷이 버려진다. 이를 packet loss라고 하며, 새로 도착한 패킷을 버릴 수도 있고, 이미 대기 중인 패킷을 버릴 수도 있다.

#### Forwarding Tables

패킷의 헤더에는 destination IP 주소가 포함되어 있다. 라우터는 패킷을 받으면 이 주소를 확인하고, **forwarding table**을 참조하여 패킷을 내보낼 outgoing link를 결정한다. Forwarding table은 destination IP 주소 또는 그 일부(prefix)를 outgoing link에 대응시킨 표이다.

Forwarding table의 내용은 **routing protocol**을 통해 자동으로 설정된다. Routing protocol은 라우터들이 경로 정보를 얻는 데 사용되며, 최단 경로처럼 프로토콜의 기준에 맞게 경로를 계산한다.

즉, forwarding은 라우터 내부에서 패킷을 알맞은 link로 보내는 동작이고, routing은 패킷이 목적지까지 이동할 경로를 결정하는 과정인 것이다.

## 2.2. Circuit Switching

Circuit switching은 packet switching과 달리 통신을 시작하기 전에 source와 destination 사이의 경로에 필요한 자원(buffer, transmission rate)을 미리 예약하는 방식이다. 전통적인 전화망이 대표적인 예이다.

Source가 데이터를 보내기 전에 네트워크는 source와 destination 사이에 연결을 확립한다. 이렇게 확립된 연결을 **circuit**이라고도 한다. Circuit이 설정되면 경로상의 각 link에서 일정한 전송률이 해당 연결에 할당된다. 할당된 자원은 연결이 유지되는 동안 다른 연결이 사용할 수 없으므로, 이 전송률을 보장할 수 있다.

예를 들어 전송률이 $R$인 link를 $n$개의 circuit이 똑같이 나누어 쓰면, 각 circuit에는 다음과 같은 전송률이 할당된다.

$$
R_{\text{circuit}} = \frac{R}{n}
$$

반면 데이터를 보내지 않는 동안에도 예약된 자원을 다른 연결이 사용할 수 없으므로, 그만큼 link 용량이 낭비될 수 있다.

#### FDM, TDM

Circuit switching에서 하나의 link는 multiplexing을 통해 여러 circuit이 함께 사용한다. 대표적인 방식으로 FDM과 TDM이 있다.

- **FDM (Frequency-Division Multiplexing)**: Link의 주파수 대역을 나누어 각 circuit에 서로 다른 주파수 대역을 할당하는 방식이다. 각 circuit은 연결이 유지되는 동안 할당받은 대역을 계속 사용한다. 이때 할당된 대역의 폭을 bandwidth라고 한다.
- **TDM (Time-Division Multiplexing)**: 시간을 frame으로 나누고, 각 frame을 다시 일정한 수의 time slot으로 나누는 방식이다. 각 circuit은 frame마다 정해진 slot을 할당받아, 그 시간에만 link의 전체 전송률로 데이터를 보낸다.

즉, FDM은 주파수를 나누어 쓰고, TDM은 시간을 나누어 쓴다.

TDM에서 link 전송률이 $R$이고 한 frame에 slot이 $n$개 있을 때, slot을 하나 할당받은 circuit의 평균 전송률은 다음과 같다.

$$
R_{\text{circuit}} = \frac{R}{n}
$$

## 2.3. Packet Switching vs Circuit Switching

| 구분 | Circuit switching | Packet switching |
|---|---|---|
| 자원 할당 | 수요와 무관하게 사전 할당 | 수요에 따라 사용 |
| 유휴 자원 | 다른 사용자가 사용 불가 | 다른 사용자가 사용 가능 |
| 전송률과 지연 | 전송률 보장, 큐잉 지연 없음 | 혼잡에 따라 큐잉 지연과 패킷 손실 발생 가능 |
| 강점 | 전송률 보장, 일정한 지연 | link 공유 효율이 높고 구현이 비교적 단순함 |

#### 사용자 수용력

전송률이 1 Mbps인 link를, 활동 상태일 때 100 kbps를 사용하고 전체 시간의 10%만 활동하는 사용자들이 공유한다고 하자.

- **Circuit switching**: 사용자마다 100 kbps를 항상 예약해야 하므로 최대 10명만 수용할 수 있다.
- **Packet switching**: 사용자가 35명이어도 동시에 11명 이상 활성 상태일 확률은 약 0.0004에 불과하다. 따라서 대부분의 시간 동안 도착 속도가 link 전송률을 넘지 않으며, 3배 이상의 사용자를 수용하면서도 circuit switching과 비슷한 성능을 낼 수 있다.

#### 혼자 전송할 때 link 전체를 사용할 수 있다

$n$명이 전송률 $R$인 link를 공유하는데, 한 명만 $D$ bits를 전송한다고 하자.

- **TDM circuit switching**: 다른 사용자의 slot이 비어 있어도 사용할 수 없으므로 $\frac{nD}{R}$초가 걸린다.
- **Packet switching**: 다른 패킷이 없으므로 link 전체를 사용할 수 있어 $\frac{D}{R}$초가 걸린다.

따라서 이 경우 packet switching이 $n$배 빠르다.

Packet switching은 지연을 예측하기 어렵다는 단점이 있지만, 데이터가 간헐적으로 발생하는 환경에서 link를 효율적으로 공유한다. 현재 통신망에는 두 방식이 모두 쓰이지만, 전체적인 흐름은 packet switching 쪽으로 이동하고 있다.